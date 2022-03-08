import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from keras.models import Model
from keras.layers import Conv2D, Input, AveragePooling2D, Dense, Reshape, Lambda
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
import keras.layers
from sklearn.datasets import fetch_lfw_people
from SpatialPyramidPooling import SpatialPyramidPooling
import imageio
from utils import InceptionBlock, rgb2gray, rgb2ycc, paper_loss, ycbcr2rgb



class ISGAN(object):
    def __init__(self):
        self.images_lfw = None
        # 基础模型
        self.enco_model = self.set_encode_model()
        self.deco_model=self.decode_model()

        # 判别器模型
        self.discriminator = self.set_discriminator()

        # 判别器编译
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy')

        # 对抗模型
        img_cover = Input(shape=(3, 256, 256))
        img_secret = Input(shape=(1, 256, 256))
        imgs_stego= self.enco_model([img_cover, img_secret])
        reconstructed_img=self.deco_model(imgs_stego)

        # 冻结判别器
        self.discriminator.trainable = False

        # 判别器决定隐写图的安全性
        security = self.discriminator(imgs_stego)

        delta = 0.001
        # 编译对抗模型，定义损失函数
        self.adversarial = Model(inputs=[img_cover, img_secret], \
                                 outputs=[imgs_stego, reconstructed_img, security])

        self.adversarial.compile(optimizer='adam', \
                                 loss=['mse', 'mse', 'binary_crossentropy'], \
                                 loss_weights=[0.5,1,0.001])

        self.adversarial.summary()

    def set_encode_model(self):
        # 定义输入
        cover_input = Input(shape=(3, 256, 256), name='cover_img')  # 载体图
        secret_input = Input(shape=(1, 256, 256), name='secret_img')  # 需要隐写的图

        # 把Y通道和CBCR通道从载体图像分离
        cover_Y = Lambda(lambda x: x[:, 0, :, :])(cover_input)
        cover_Y = Reshape((1, 256, 256), name="cover_img_Y")(cover_Y)

        cover_cc = Lambda(lambda x: x[:, 1:, :, :])(cover_input)
        cover_cc = Reshape((2, 256, 256), name="cover_img_cc")(cover_cc)

        # 把载体图的Y通道和隐写图链接
        combined_input = keras.layers.concatenate([cover_Y, secret_input], axis=1)

        # 编码器
        L1 = Conv2D(16, 3, padding='same', data_format='channels_first')(combined_input)
        L1 = BatchNormalization(momentum=0.9)(L1)
        L1 = LeakyReLU(alpha=0.2)(L1)

        L2 = InceptionBlock(16, 32)(L1)
        L3 = InceptionBlock(32, 64)(L2)
        L4 = InceptionBlock(64, 128)(L3)
        L5 = InceptionBlock(128, 256)(L4)
        L6 = InceptionBlock(256, 128)(L5)
        L7 = InceptionBlock(128, 64)(L6)
        L8 = InceptionBlock(64, 32)(L7)

        L9 = Conv2D(16, 3, padding='same', data_format='channels_first')(L8)
        L9 = BatchNormalization(momentum=0.9)(L9)
        L9 = LeakyReLU(alpha=0.2)(L9)

        enc_Y_output = Conv2D(1, 1, padding='same', activation='tanh', name="enc_Y_output",
                              data_format='channels_first')(L9)
        enc_output = keras.layers.concatenate([enc_Y_output, cover_cc], axis=1, name="enc_output")


        # 建立模型，输入是YCBCR的载体图和灰度隐写图
        # 输出是YCBCR的重构图和重构图的灰度图
        model = Model(inputs=[cover_input, secret_input], outputs=enc_output)
        model.summary()
        return model

    def decode_model(self):
        # 解码器
        depth = 32
        enc_Y_output=Input(shape=(3, 256, 256), name='stego')
        cover_Y = Lambda(lambda x: x[:, 0, :, :])(enc_Y_output)
        cover_Y = Reshape((1, 256, 256), name="cover_img_Y")(cover_Y)
        L1 = Conv2D(depth, 3, padding='same', data_format='channels_first')(cover_Y)
        L1 = BatchNormalization(momentum=0.9)(L1)
        L1 = LeakyReLU(alpha=0.2)(L1)

        L2 = Conv2D(depth * 2, 3, padding='same', data_format='channels_first')(L1)
        L2 = BatchNormalization(momentum=0.9)(L2)
        L2 = LeakyReLU(alpha=0.2)(L2)

        L3 = Conv2D(depth * 4, 3, padding='same', data_format='channels_first')(L2)
        L3 = BatchNormalization(momentum=0.9)(L3)
        L3 = LeakyReLU(alpha=0.2)(L3)

        L4 = Conv2D(depth * 2, 3, padding='same', data_format='channels_first')(L3)
        L4 = BatchNormalization(momentum=0.9)(L4)
        L4 = LeakyReLU(alpha=0.2)(L4)

        L5 = Conv2D(depth, 3, padding='same', data_format='channels_first')(L4)
        L5 = BatchNormalization(momentum=0.9)(L5)
        L5 = LeakyReLU(alpha=0.2)(L5)

        dec_output = Conv2D(1, 1, padding='same', activation='sigmoid', name="dec_output",
                            data_format='channels_first')(L5)

        model = Model(inputs=enc_Y_output, outputs=dec_output)
        return model

    def set_discriminator(self):
        img_input = Input(shape=(3, 256, 256), name='discrimator_input')
        L1 = Conv2D(8, 3, padding='same', data_format='channels_first')(img_input)
        L1 = BatchNormalization(momentum=0.9)(L1)
        L1 = LeakyReLU(alpha=0.2)(L1)
        L1 = AveragePooling2D(pool_size=5, strides=2, padding='same', data_format='channels_first')(L1)
        L2 = Conv2D(16, 3, padding='same', data_format='channels_first')(L1)
        L2 = BatchNormalization(momentum=0.9)(L2)
        L2 = LeakyReLU(alpha=0.2)(L2)

        L2 = AveragePooling2D(pool_size=5, strides=2, padding='same', data_format='channels_first')(L2)

        L3 = Conv2D(32, 1, padding='same', data_format='channels_first')(L2)
        L3 = BatchNormalization(momentum=0.9)(L3)
        L3 = AveragePooling2D(pool_size=5, strides=2, padding='same', data_format='channels_first')(L3)

        L4 = Conv2D(64, 1, padding='same', data_format='channels_first')(L3)
        L4 = BatchNormalization(momentum=0.9)(L4)
        L4 = AveragePooling2D(pool_size=5, strides=2, padding='same', data_format='channels_first')(L4)

        L5 = Conv2D(128, 3, padding='same', data_format='channels_first')(L4)
        L5 = BatchNormalization(momentum=0.9)(L5)
        L5 = LeakyReLU(alpha=0.2)(L5)
        L5 = AveragePooling2D(pool_size=5, strides=2, padding='same', data_format='channels_first')(L5)

        L6 = SpatialPyramidPooling([1, 2, 4])(L5)
        L7 = Dense(128)(L6)
        L8 = Dense(1, activation='tanh', name="D_output")(L7)

        discriminator = Model(inputs=img_input, outputs=L8)
        discriminator.summary()

        return discriminator

    def train(self, epochs, batch_size=4):
        # 加载LFW数据
        print("可能需要几分钟加载LFW数据")
        # Smaller dataset used for implementation evaluation
        lfw_people = fetch_lfw_people(color=True, resize=1.0, \
                                      slice_=(slice(0, 250), slice(0, 250)), \
                                      min_faces_per_person=3)

        images_rgb = lfw_people.images
        images_rgb = np.moveaxis(images_rgb, -1, 1)  # 只选取部分数据，否则显存撑不住
        images_rgb = images_rgb[:2000]
        print(type(images_rgb))
        print(images_rgb.shape)

        # 对LFW图像进行PAD操作
        images_rgb = np.pad(images_rgb, ((0, 0), (0, 0), (3, 3), (3, 3)), 'constant')
        self.images_lfw = images_rgb

        # 把载体图从RGB转换成YCBCR灰度图
        images_ycc = np.zeros(images_rgb.shape)
        secret_gray = np.zeros((images_rgb.shape[0], 1, images_rgb.shape[2], images_rgb.shape[3]))
        for k in range(images_rgb.shape[0]):
            EncryptionImg = np.zeros((3, 256, 256), np.uint8)

            images_ycc[k, :, :, :] = rgb2ycc(images_rgb[k, :, :, :])
            secret_gray[k, 0, :, :] = rgb2gray(images_rgb[k, :, :, :])

        X_train_ycc = (images_ycc.astype(np.float32) - 127.5) / 127.5
        X_train_gray = (secret_gray.astype(np.float32) - 127.5) / 127.5

        original = np.ones((batch_size, 1))
        encrypted = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # 从载体图随机选择一个批量
            idx = np.random.randint(0, X_train_ycc.shape[0], batch_size)
            imgs_cover = X_train_ycc[idx]
            # print(idx)

            # 从隐写图随机写入一个批量
            idx = np.random.randint(0, X_train_ycc.shape[0], batch_size)
            imgs_gray = X_train_gray[idx]
            # print(idx)

            # 生成
            imgs_stego= self.enco_model.predict([imgs_cover, imgs_gray])
            sec=self.deco_model(imgs_stego)

            # 训练判别器
            d_loss_real = self.discriminator.train_on_batch(imgs_cover, original)
            d_loss_encrypted = self.discriminator.train_on_batch(imgs_stego, encrypted)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_encrypted)

            g_loss = self.adversarial.train_on_batch([imgs_cover, imgs_gray], [imgs_cover, imgs_gray, original])

            # 输出训练的loss
            print("{} [D loss: {}] [G loss: {}]".format(epoch, d_loss, g_loss[0]))

            self.adversarial.save('./model/adversarial.h5')
            self.discriminator.save('./model/discriminator.h5')
            self.enco_model.save('./model/enco_model.h5')
            self.deco_model.save('./model/deco_model.h5')
            if epoch % 10 == 0:
                self.draw_images(1, epoch)

    # 输出图像效果
    def draw_images(self, nb_images=1, epoch=1):
        # 随机选择数据

        cover_idx = np.random.randint(0, self.images_lfw.shape[0], nb_images)
        secret_idx = np.random.randint(0, self.images_lfw.shape[0], nb_images)
        imgs_cover = self.images_lfw[cover_idx]
        imgs_secret = self.images_lfw[secret_idx]

        print(imgs_cover.shape)
        print(type(imgs_cover))
        images_ycc = np.zeros(imgs_cover.shape)
        secret_gray = np.zeros((imgs_secret.shape[0], 1, imgs_cover.shape[2], imgs_cover.shape[3]))

        # 载体图和隐写图转换为灰度图
        for k in range(nb_images):
            images_ycc[k, :, :, :] = rgb2ycc(imgs_cover[k, :, :, :])
            secret_gray[k, 0, :, :] = rgb2gray(imgs_secret[k, :, :, :])

        X_test_ycc = (images_ycc.astype(np.float32) - 127.5) / 127.5
        X_test_gray = (secret_gray.astype(np.float32) - 127.5) / 127.5

        imgs_stego= self.enco_model.predict([X_test_ycc, X_test_gray])
        imgs_recstr=self.deco_model.predict(imgs_stego)

        imgs_stego = imgs_stego.astype(np.float32) * 127.5 + 127.5
        imgs_recstr = imgs_recstr.astype(np.float32) * 127.5 + 127.5

        imgs_cover = imgs_cover.transpose((0, 2, 3, 1))
        imgs_stego = imgs_stego.transpose((0, 2, 3, 1))
        secret_gray = np.reshape(secret_gray, (nb_images, 256, 256))
        imgs_recstr = np.reshape(imgs_recstr, (nb_images, 256, 256))

        for k in range(nb_images):
            imageio.imwrite('images/{}_{}_cover.png'.format(epoch, k), imgs_cover[k, :, :, :])
            imageio.imwrite('images/{}_{}_secret.png'.format(epoch, k), secret_gray[k, :, :])
            imageio.imwrite('images/{}_{}_stego.png'.format(epoch, k), imgs_stego[k, :, :, :])
            imageio.imwrite('images/{}_{}_recstr.png'.format(epoch, k), imgs_recstr[k, :, :])
            ycbcr_image = imageio.imread('images/{}_{}_stego.png'.format(epoch, k))
            cycle_image = ycbcr2rgb(ycbcr_image)
            imageio.imwrite('images/{}_{}_stego_rgb.png'.format(epoch, k), cycle_image)

        print("测试图像已写入")


if __name__ == "__main__":
    is_model = ISGAN()
    is_model.train(epochs=10000)