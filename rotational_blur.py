# -*- coding: UTF-8 -*-
# @Time    : 22/03/2019 17:43
# @Author  : QYD


import numpy as np
from utils import cal_r_max, median_draw_circle, convolution_image
import cv2 as cv


def rotation_blur(img: np.array, theta, center=None):
    """
    利用中点画圆法
    :param img: 输入图像
    :param theta: 曝光时间内，图像旋转的角度。
    :param center:旋转中心的坐标，默认位置在图像中心。
    :return: 模糊后的图像。
    """

    height, width, _ = img.shape
    if not center:
        center = width // 2, height // 2
    r_max = cal_r_max(img, center)
    circle_list = median_draw_circle(r_max)
    img_blur = convolution_image(img=img, center=center, circle_list=circle_list, theta=theta, mode="blur")
    return img_blur


if __name__ == '__main__':
    img = cv.imread("samples/origin.jpg")
    img_rotational_blur = rotation_blur(img=img, theta=0.20)
    cv.imwrite("samples/origin_blur_0.20.jpg", img_rotational_blur)
