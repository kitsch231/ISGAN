import numpy as np
import imageio
import matplotlib.pyplot as plt
from utils import rgb2ycc,rgb2gray
from keras import models
nb_images=1
enco_model=models.load_model('./model/enco_model.h5')
deco_model=models.load_model('./model/deco_model.h5')

imgs_cover = imageio.imread("./images/230_0_cover.png").transpose((2,0,1))
print(imgs_cover.shape)
imgs_secret=imageio.imread("./images/180_0_cover.png").transpose((2,0,1))


print(imgs_cover.shape)
images_ycc = np.zeros(imgs_cover.shape).reshape((1,3,256,256))
secret_gray = np.zeros((1,imgs_cover.shape[1], imgs_cover.shape[2])).reshape((1,1,256,256))
# 载体图和隐写图转换为灰度图
print(rgb2gray(imgs_secret).shape)
images_ycc[0,:, :, :] = rgb2ycc(imgs_cover[:, :, :])
secret_gray[0,0,:, :] = rgb2gray(imgs_secret)

X_test_ycc = (images_ycc.astype(np.float32) - 127.5) / 127.5
X_test_gray = (secret_gray.astype(np.float32) - 127.5) / 127.5
print(X_test_ycc.shape)
print(X_test_gray.shape)
imgs_stego = enco_model.predict([X_test_ycc, X_test_gray])
imgs_recstr = deco_model.predict(imgs_stego)

imgs_stego = imgs_stego.astype(np.float32) * 127.5 + 127.5
imgs_recstr = imgs_recstr.astype(np.float32) * 127.5 + 127.5

# imgs_cover = imgs_cover.transpose((0, 2, 3, 1))
# print(imgs_cover.shape)
imgs_stego = imgs_stego.transpose((0, 2, 3, 1))
secret_gray = np.reshape(secret_gray, (nb_images, 256, 256))
imgs_recstr = np.reshape(imgs_recstr, (nb_images, 256, 256))


imageio.imwrite('1.png',imgs_stego[0,:,:,:])
#imageio.imwrite('2.png',imgs_recstr[:, :])
plt.imshow(imgs_recstr[0,:,:])
plt.show()
