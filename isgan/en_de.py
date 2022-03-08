import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
'''
加密函数
img_encrypt:原始图像(二维ndarray)
key:密钥列表，大小为7(1、2、3为混沌系统初始条件；4、5、6为分岔参数u，7为重复加密次数)
return:返回加密后的图像
'''

def encrypt(img,key):
  #图像的宽高
  [w,h]=[256,256]
  #混沌系统初始条件
  x1=key[0]
  x2=key[1]
  x3=key[2]
  #分岔参数u
  u1=key[3]
  u2=key[4]
  u3=key[5]
  #加密次数
  n=key[6]
  #一个临时数组用于返回加密后的图像，可以不影响原始图像
  img_tmp=np.zeros((w,h))
  #对原始图像的每个像素都处理n次
  for k in range(n):
    for i in range(w):
      for j in range(h):
        #计算混沌序列值
        x1=u1*x1*(1-x1)
        x2=u2*x2*(1-x2)
        x3=u3*x3*(1-x3)
        ##混沌值位于[0,1]区间内，所以可以看做是一个系数，乘以最大灰度值并转成整数用于异或运算即可
        r1=int(x1*255)
        r2=int(x2*255)
        r3=int(x3*255)
        img_tmp[i][j]=(((r1+r2)^r3)+img[i][j])%256
    #下一轮加密重新初始化混沌系统
    x1=key[0]
    x2=key[1]
    x3=key[3]
  return img_tmp


'''
解密函数
img:加密后图像(二维ndarray)
key:密钥列表，大小为7(1、2、3为混沌系统初始条件；4、5、6为分岔参数u，7为重复加密次数)
return:返回解密后的图像
'''
def decrypt(img,key):
    #图像的宽高
  [w,h]=[256,256]
  #混沌系统初始条件
  x1=key[0]
  x2=key[1]
  x3=key[2]
  #分岔参数u
  u1=key[3]
  u2=key[4]
  u3=key[5]
  #加密次数
  n=key[6]
  #一个临时数组用于返回加密后的图像，可以不影响传入的加密图像
  img_tmp=np.zeros((w,h))
  #对原始图像的每个像素都处理n次
  for k in range(n):
    for i in range(w):
      for j in range(h):
        x1=u1*x1*(1-x1)
        x2=u2*x2*(1-x2)
        x3=u3*x3*(1-x3)
        r1=int(x1*255)
        r2=int(x2*255)
        r3=int(x3*255)
        img_tmp[i][j]=(img[i][j]-((r1+r2)^r3))%256
        # img[i][j]=(img[i][j]-((r1[i*w+j]+r2[i*w+j])^r3[i*w+j]))%256
    #下一轮加密重新初始化混沌系统
    x1=key[0]
    x2=key[1]
    x3=key[3]
  return img_tmp


def main():
    # 原始图像路径
    path = './images/0_0_secret.png'
    # 加密密钥参数列表
    key = [0.343, 0.432, 0.63, 3.769, 3.82, 3.85, 1]
    # 读取原始图像
    img = cv2.imread(path)
    # img.shape=>(512,512,3),宽、高、通道数量

    # 图像灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 混沌加密原始图像
    img_encrypt = encrypt(img_gray, key)
    # 错误密钥解密图像
    a=np.random.randint(1,20,(256,256))
    print(img_encrypt)
    img_encrypt=img_encrypt-a
    print(img_encrypt)
    wrong_key = [0.342, 0.432, 0.61, 3.769, 3.82, 3.85, 1]
    wrong_decrypt = decrypt(img_encrypt, wrong_key)
    # 正确密钥解密图像
    img_decrypt = decrypt(img_encrypt, key)

    # 结果展示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
    # 子图1，原始图像
    plt.subplot(221)
    # plt默认使用三通道显示图像，所以需要制定cmap参数为gray
    # imshow()对图像进行处理，画出图像，show()进行图像显示
    plt.imshow(img_gray, cmap='gray')
    plt.title('原始图像(灰度化)')
    # 不显示坐标轴
    plt.axis('off')

    # 子图2，加密后图像
    plt.subplot(222)
    plt.imshow(img_encrypt, cmap='gray')
    plt.title('加密图像(密钥{}'.format(key))
    plt.axis('off')

    # 子图3，错误密钥解密结果
    plt.subplot(223)
    plt.imshow(wrong_decrypt, cmap='gray')
    plt.title('解密图像(密钥{})'.format(wrong_key))
    plt.axis('off')

    # 子图4，正确密钥解密结果
    plt.subplot(224)
    plt.imshow(img_decrypt, cmap='gray')
    plt.title('解密图像(密钥{})'.format(key))
    plt.axis('off')

    # 设置子图默认的间距
    # plt.tight_layout()
    # 保存图像
    # cv2.imwrite('./lean_original.jpg',img)
    # cv2.imwrite('./lean_gray.jpg',img_gray)
    # cv2.imwrite('./lean_encrypt.jpg',img_encrypt)
    # cv2.imwrite('./lean_gray.jpg',wrong_decrypt)
    # cv2.imwrite('./lean_decrypt.jpg',img_decrypt)

    # 显示图像
    plt.show()


if __name__ == '__main__':
  main()

