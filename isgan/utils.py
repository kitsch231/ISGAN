import numpy as np
import scipy

from keras.layers import Input, Conv2D, MaxPooling2D, Activation
from keras.layers import *
from keras.models import Model
import keras.layers

# Utility functions for computing losses

# Luminance
def L(x, y, C1=1):
    mu_x, mu_y = np.mean(x), np.mean(y)
    return (2 * mu_x * mu_y + C1) / (mu_x**2 + mu_y**2 + C1)

# Contrast
def C(x, y, C2=1):
    theta_x, theta_y = np.std(x), np.std(y)
    return (2 * theta_x * theta_y + C2) / (theta_x**2 + theta_y**2 + C2)

# Structure
def S(x, y, C3=1):
    theta_x, theta_y, theta_xy = np.std(x), np.std(y), np.cov(x, y)[0, 1]
    return (theta_xy + C3) / (theta_x * theta_y + C3)

# MSE loss
def MSE(x, y):
    return np.mean(x**2 - y**2)

# SSIM loss function between two images
def SSIM(x, y, alpha=1, beta=1, gamma=1):
    return L(x, y)**alpha * C(x, y)**beta * S(x, y)**gamma

# Multimodal SSIM loss function between two (channel first !) images 
def MSSIM(x, y, M=5):
    result = 0
    x_k, y_k = x, y
    for _ in range(M):
        result *= C(x_k, y_k) * S(x_k, y_k)
        # Downsample by 2 the images
        x_k = scipy.signal.decimate(x_k, 2, axis=-1)
        x_k = scipy.signal.decimate(x_k, 2, axis=-2)
        y_k = scipy.signal.decimate(y_k, 2, axis=-1)
        y_k = scipy.signal.decimate(y_k, 2, axis=-2)
    
    result *= L(x_k, y_k) * C(x_k, y_k) * S(x_k, y_k)
    return result

# Loss function defined in the paper for two images compatible with keras
def paper_loss(alpha=0.5, beta=0.3):
    def loss(y_true, y_pred):
        return alpha * (1 - SSIM(y_true, y_pred)) + (1 - alpha) * (1 - MSSIM(y_true, y_pred)) \
               + beta * MSE(y_true, y_pred)
    return loss


# Color space conversions

def rgb2ycc(img_rgb):
    # np.array([[0.299, 0.587, 0.114],
    #           [-0.1687 - 0.3313, 0.5],
    #           [0.5, -0.4187, 0.0813]])

    output = np.zeros(np.shape(img_rgb))
    output[0, :, :] = 0.299 * img_rgb[0, :, :] + 0.587 * img_rgb[1, :, :] + 0.114 * img_rgb[2, :, :]
    output[1, :, :] = -0.1687 * img_rgb[0, :, :] - 0.3313 * img_rgb[1, :, :] \
                      + 0.5 * img_rgb[2, :, :] + 128
    output[2, :, :] = 0.5 * img_rgb[0, :, :] - 0.4187 * img_rgb[1, :, :] \
                      + 0.0813 * img_rgb[2, :, :] + 128
    return output

def rgb2ycbcr(rgb_image):
    """convert rgb into ycbcr"""
    if len(rgb_image.shape)!=3 or rgb_image.shape[2]!=3:
        raise ValueError("input image is not a rgb image")
    rgb_image = rgb_image.astype(np.float32)
    # 1：创建变换矩阵，和偏移量
    transform_matrix =np.array([[0.299, 0.587, 0.114],
              [-0.1687,- 0.3313, 0.5],
              [0.5, -0.4187, 0.0813]])
    shift_matrix = np.array([16, 128, 128])
    ycbcr_image = np.zeros(shape=rgb_image.shape)
    w, h, _ = rgb_image.shape
    # 2：遍历每个像素点的三个通道进行变换
    for i in range(w):
        for j in range(h):
            ycbcr_image[i, j, :] = np.dot(transform_matrix, rgb_image[i, j, :]) + shift_matrix
    return ycbcr_image



def ycbcr2rgb(ycbcr_image):
    """convert ycbcr into rgb"""
    if len(ycbcr_image.shape)!=3 or ycbcr_image.shape[2]!=3:
        raise ValueError("input image is not a rgb image")
    ycbcr_image = ycbcr_image.astype(np.float32)
    transform_matrix =np.array([[0.299, 0.587, 0.114],
              [-0.1687,- 0.3313, 0.5],
              [0.5, -0.4187, 0.0813]])
    transform_matrix_inv = np.linalg.inv(transform_matrix)
    shift_matrix = np.array([16, 128, 128])
    rgb_image = np.zeros(shape=ycbcr_image.shape)
    w, h, _ = ycbcr_image.shape
    for i in range(w):
        for j in range(h):
            rgb_image[i, j, :] = np.dot(transform_matrix_inv, ycbcr_image[i, j, :]) - np.dot(transform_matrix_inv, shift_matrix)
    return rgb_image


def rgb2gray(img_rgb):
    """
    Transform a RGB image into a grayscale one using weighted method. Shape: channels first.
    """
    output = np.zeros((1, img_rgb.shape[1], img_rgb.shape[2]))
    output[0, :, :] = 0.3 * img_rgb[0, :, :] + 0.59 * img_rgb[1, :, :] + 0.11 * img_rgb[2, :, :]
    return output

def InceptionBlock(filters_in, filters_out):
    input_layer = Input(shape=(filters_in, 256, 256))
    tower_filters = int(filters_out / 4)

    tower_1 = Conv2D(tower_filters, 1, padding='same', activation='relu', data_format='channels_first')(input_layer)

    tower_2 = Conv2D(tower_filters, 1, padding='same', activation='relu', data_format='channels_first')(input_layer)
    tower_2 = Conv2D(tower_filters, 3, padding='same', activation='relu', data_format='channels_first')(tower_2)

    tower_3 = Conv2D(tower_filters, 1, padding='same', activation='relu', data_format='channels_first')(input_layer)
    tower_3 = Conv2D(tower_filters, 5, padding='same', activation='relu', data_format='channels_first')(tower_3)

    tower_4 = MaxPooling2D(tower_filters, padding='same', strides=(1, 1), data_format='channels_first')(input_layer)
    tower_4 = Conv2D(tower_filters, 1, padding='same', activation='relu', data_format='channels_first')(tower_4)

    concat = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=1)
    #concat=Reshape((16,256,32))(concat)

    res_link = Conv2D(filters_out, 1, padding='same', activation='relu', data_format='channels_first')(input_layer)

    output = keras.layers.add([concat, res_link])
    output = Activation('relu')(output)

    model_output = Model([input_layer], output)
    return model_output