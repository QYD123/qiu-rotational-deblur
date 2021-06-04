# -*- coding: UTF-8 -*-
# @Time    : 22/04/2019 17:43
# @Author  : QYD

import numpy as np
from numpy import fft
import copy
import cv2 as cv
from math import sqrt


def inverse(psf_seq, blur_seq):
    """
    输入：点扩散函数（离散序列），退化后的序列（离散序列）。
    输出：原图像序列。
    算法：逆滤波算法。由于退化矩阵的不可逆（病态性）的特性，因此无法直接的使用。因此常常和对角加载等技术相结合。
    """
    psf_seq_pinyu = fft.fft(psf_seq)
    blur_seq_pinyu = fft.fft(blur_seq)
    deblur_seq = blur_seq_pinyu / psf_seq_pinyu
    deblur_seq = np.abs(fft.ifft(deblur_seq))
    deblur_seq = deblur_seq.astype(np.uint8)
    return deblur_seq


def wiener(psf_seq, blur_seq):
    """
       输入：点扩散函数（离散序列），退化后的序列（离散序列）。
       输出：原图像序列。
       算法：维纳滤波算法（最小均方差滤波）。其目标是使恢复的图像和未退化的图像之间的均方差最小。
       纠正了逆滤波的病态性。要求估计得到未退化的图像的噪声和功率谱。
       问题：①空间域的循环卷积等同于傅里叶变化之后频域的乘法。
            ②（普通）卷积等同于频域的怎样的变化？首先怎样定义普通卷积？是‘same’，‘full','valid"中的哪一种形式？
                冈萨雷斯的书上的卷积指的是先‘full'卷积，再裁剪到原图大小。这种卷积的频域的形式该怎样写？
    """
    psf_seq_pinyu = fft.fft(psf_seq)
    blur_seq_pinyu = fft.fft(blur_seq)
    num = (np.abs(psf_seq_pinyu) ** 2) * blur_seq_pinyu
    den = (np.abs(psf_seq_pinyu) ** 2 + 0.001) * psf_seq_pinyu
    deblur_seq = np.abs(fft.ifft(num / den))  # wiener滤波的分母可能是零值，这会导致除法出现问题，对角加载可以解决这种问题。
    deblur_seq = deblur_seq.astype(np.uint8)
    return deblur_seq


def psf(psf_seq, seq):
    """
    核序列和像素序列的循环卷积。空间域的循环卷积相当于频域的阵列相乘。
    :param psf_seq: 核序列
    :param seq: 像素序列
    :return: 得到的序列。
    """
    psf_seq_pinyu = fft.fft(psf_seq)
    seq_pinyu = fft.fft(seq)
    blur_seq_pinyu = psf_seq_pinyu * seq_pinyu
    blur_seq = fft.ifft(blur_seq_pinyu)
    blur_seq = np.abs(blur_seq)
    blur_seq = blur_seq.astype(np.uint8)
    return blur_seq


def cal_r_max(img, center=None):
    """

    :param img: the input image
    :param center: the rotation center of the img,default is (width//2,height//2)
    :return: r_max
    """
    height, width, _ = img.shape
    if not center:
        center_x = width // 2
        center_y = height // 2
    else:
        center_x, center_y = center
    r1 = int(sqrt(center_x ** 2 + center_y ** 2))
    r2 = int(sqrt((center_x - width + 1) ** 2 + center_y ** 2))
    r3 = int(sqrt(center_x ** 2 + (center_y - height + 1) ** 2))
    r4 = int(sqrt((center_x - width + 1) ** 2 + (center_y - height + 1) ** 2))
    r_max = max(r1, r2, r3, r4)
    return r_max


def median_draw_circle(r_max):
    """
    draw circle using the median draw method
    :parameter r_max:the max radius of the circle
    :return:the list whose each row contains the cord of the circle in given radius
    """
    circle_list = []
    for r in range(1, r_max):
        seq = []
        x, y, d = 0, r, 1.25 - r
        seq.append([x, y])
        while x < y:  # 获得第一象限上的圆弧的像素序列的坐标。
            if d < 0:
                d = d + 2 * x + 3
            else:
                d = d + 2 * (x - y) + 5
                y = y - 1
            x = x + 1
            seq.append([x, y])
        # 将画得到的8分之1个圆的坐标映射到整个圆上
        seq_change = list(map(change1, seq))
        seq_change.reverse()
        seq += seq_change
        seq_change = list(map(change2, seq))
        seq_change.reverse()
        seq += seq_change
        seq_change = list(map(change3, seq))
        seq_change.reverse()
        seq += seq_change
        circle_list.append(seq)
    return circle_list


def change1(a):
    b = [0, 0]
    b[0], b[1] = a[1], a[0]
    return b


def change2(a):
    b = [0, 0]
    b[0], b[1] = a[0], -a[1]
    return b


def change3(a):
    b = [0, 0]
    b[0], b[1] = -a[0], a[1]
    return b


def convolution_image(img, center, circle_list: list, theta, mode="blur"):
    """

    calculate psf_list of the circle_list
    :parameter theta:the rotation theta of the image
    :parameter method:the calculate mode of the psf_mode
    """
    height, width, _ = img.shape
    if not center:
        center_x = width // 2
        center_y = height // 2
    else:
        center_x, center_y = center
    if mode == "blur":
        img_out = np.zeros_like(img)
    elif mode == "wiener" or mode == "dia_conv":
        img_out = copy.deepcopy(img)
    for i in range(len(circle_list)):
        r = i + 1
        num = len(circle_list[i])
        a = 1 + int(r * theta)
        pix_seq_b = []
        pix_seq_g = []
        pix_seq_r = []
        for cord in circle_list[i]:
            x, y = cord[0] + center_x, - cord[1] + center_y
            # 处理四个角上的情况。
            if x < 0:
                x = 0
            if x > width - 1:
                x = width - 1
            if y < 0:
                y = 0
            if y > height - 1:
                y = height - 1
            pix_seq_b.append(img[y][x][0])
            pix_seq_g.append(img[y][x][1])
            pix_seq_r.append(img[y][x][2])
            cord[0], cord[1] = x, y
        if mode == "blur":
            psf_seq = [1 / a] * a + (num - a) * [0]
            method = psf
        elif mode == "wiener":
            psf_seq = [1 / a] * a + (num - a) * [0]
            method = wiener
        elif mode == "dia_conv":
            psf_seq = [2 * (1 / a)] + [1 / a] * (a - 1) + (num - a) * [0]
            method = inverse
        pix_seq_b_deblur = method(psf_seq, pix_seq_b)
        pix_seq_g_deblur = method(psf_seq, pix_seq_g)
        pix_seq_r_deblur = method(psf_seq, pix_seq_r)
        for pix_index in range(len(pix_seq_b_deblur)):
            x, y = circle_list[i][pix_index]
            img_out[y][x] = pix_seq_b_deblur[pix_index], pix_seq_g_deblur[pix_index], pix_seq_r_deblur[pix_index]
    if mode == "blur":
        img_out = cv.medianBlur(img_out, ksize=3)
    return img_out
