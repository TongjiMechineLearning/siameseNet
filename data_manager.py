import cv2
import numpy as np
import os
from math import *
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import config
from scipy import *
from _thread import *
import time

def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name

def Get_All_Data():
    folder_data = config.img_data_path
    list_name = []
    for path in folder_data:
        list_name = listdir(path, list_name)
    return list_name

def random_crop_image(image):

    height, width = image.shape[:2]

    w = np.random.randint(width//5, width - 2, size=1)[0]
    h = np.random.randint(height//5, height - 2, size=1)[0]

    x = np.random.randint(0, width - w - 2, size=1)[0]
    y = np.random.randint(0, height - h - 2, size=1)[0]

    image_crop = image[y:h + y, x:w + x, ...]

    return image_crop

def rotate_image(img, angle, img_w, img_h, scale=1.0):
    # 获取图像尺寸
    # sp = img.shape
    width = img_w
    height = img_h
    # 执行旋转
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.5)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

    # center = (width / 2, height / 2)
    # # 执行旋转
    # M = cv2.getRotationMatrix2D(center, angle, scale)
    # rotated = cv2.warpAffine(img, M, (width, height), borderValue=(255, 255, 255))

    # 返回旋转后的图像
    return imgRotation

#定义添加椒盐噪声的函数
def SaltAndPepper(src,percetage):
    SP_NoiseImg=src
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(SP_NoiseNum):
        randX=random.random_integers(0,src.shape[0]-1)
        randY=random.random_integers(0,src.shape[1]-1)
        if random.random_integers(0,1) == 0:
            SP_NoiseImg[randX,randY] = 0
        else:
            SP_NoiseImg[randX,randY] = 255
    return SP_NoiseImg

#定义添加高斯噪声的函数
def addGaussianNoise(image,percetage):
    G_Noiseimg = image
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, image.shape[0])
        temp_y = np.random.randint(0, image.shape[1])
        G_Noiseimg[temp_x][temp_y] = 255
    return G_Noiseimg

def Image_Data_Generator(ori_img):

    img = random_crop_image(ori_img)
    angle = np.random.randint(360)

    if np.random.randint(10) > 5:
        img = rotate_image(img, angle, img.shape[1], img.shape[0])

    randval = np.random.randint(10)

    if randval < 3:
        val = np.random.randint(20) / 10.0
    elif 3 <= randval < 7:
        val = np.random.randint(5) / 10.0
    else:
        val = 1

    img = exposure.adjust_gamma(img, val)  # 调暗

    if np.random.randint(10) > 5:
        img = addGaussianNoise(img, 0.01)  # 添加10%的高斯噪声
    elif np.random.randint(10) > 5:
        img = SaltAndPepper(img, 0.01)  # 再添加10%的椒盐噪声

    return img

if __name__ == '__main__':

    folder_data = config.img_data_path
    img_new_path = config.img_new_path


    list_name = []
    for path in folder_data:
        list_name = listdir(path, list_name)

    idx = 0
    for img_path in list_name:


        print(img_path)

        img_data = cv2.imread(img_path)

        if img_data is None:
            continue

        if not os.path.exists("{}/{}".format(img_new_path, idx//100)):
            os.makedirs("{}/{}".format(img_new_path, idx//100))

        if not os.path.exists("{}/{}/{}".format(img_new_path, idx//100, idx%100)):
            os.makedirs("{}/{}/{}".format(img_new_path,idx//100, idx%100))

        for i in range(10):
            img = random_crop_image(img_data)
            # if np.random.randint(10) > 5:
            #
            #     if np.random.randint(10) > 7:
            #         img = addGaussianNoise(img, 0.05)  # 添加10%的高斯噪声
            #     if np.random.randint(10) > 7:
            #         img = SaltAndPepper(img, 0.05)  # 再添加10%的椒盐噪声

            cv2.imwrite("{}/{}/{}/r_{}.jpg".format(img_new_path, idx//100, idx%100, i), cv2.resize(img,(224,224)))

        for i in range(5):
            angle = np.random.randint(45)
            img = rotate_image(img, angle, img.shape[1], img.shape[0])

            # if np.random.randint(10) > 5:
            #
            #     if np.random.randint(10) > 7:
            #         img = addGaussianNoise(img, 0.05) # 添加10%的高斯噪声
            #     if np.random.randint(10) > 7:
            #         img = SaltAndPepper(img, 0.05) # 再添加10%的椒盐噪声

            cv2.imwrite("{}/{}/{}/a_{}.jpg".format(img_new_path,idx//100, idx%100, i), cv2.resize(img,(224,224)))

        for i in range(1):
            # randval = np.random.randint(10)
            #
            if np.random.randint(10) > 7:
                img = addGaussianNoise(img_data, 0.05)  # 添加10%的高斯噪声
            else:
                img = SaltAndPepper(img_data, 0.05)  # 再添加10%的椒盐噪声

            cv2.imwrite("{}/{}/{}/n_{}.jpg".format(img_new_path,idx//100, idx%100, i), cv2.resize(img, (224, 224)))

        idx += 1


