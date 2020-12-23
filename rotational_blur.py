# -*- coding: UTF-8 -*-
# @Time    : 22/03/2019 17:43
# @Author  : QYD


from math import sqrt
import numpy as np
from api import psf
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
        center_x = width // 2
        center_y = height // 2
    else:
        center_x, center_y = center
    img_blur = np.zeros_like(img)

    r1 = int(sqrt(center_x ** 2 + center_y ** 2))
    r2 = int(sqrt((center_x - width + 1) ** 2 + center_y ** 2))
    r3 = int(sqrt(center_x ** 2 + (center_y - height + 1) ** 2))
    r4 = int(sqrt((center_x - width + 1) ** 2 + (center_y - height + 1) ** 2))
    r_max = max(r1, r2, r3, r4)

    # 用中点画圆法提取旋转中心的距离为r的像素序列，将其和模糊核做循环卷积，得到模糊序列。
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

        num = len(seq)
        a = 1 + int(r * theta)
        psf_seq = [1 / a] * a + (num - a) * [0]  # 模糊核
        pix_seq_b = []
        pix_seq_g = []
        pix_seq_r = []
        for i in seq:
            # 由画圆坐标系下的坐标和旋转中心的坐标得到像素坐标系下点的坐标。
            i[0], i[1] = i[0] + center_x, - i[1] + center_y
            # 扩充四个角上的像素序列，当圆弧超出图片范围时，用
            if i[0] < 0:
                i[0] = 0
            if i[0] > width - 1:
                i[0] = width - 1
            if i[1] < 0:
                i[1] = 0
            if i[1] > height - 1:
                i[1] = height - 1
            pix_seq_b.append(img[i[1]][i[0]][0])
            pix_seq_g.append(img[i[1]][i[0]][1])
            pix_seq_r.append(img[i[1]][i[0]][2])
        pix_seq_b_blur = psf(psf_seq, pix_seq_b)
        pix_seq_g_blur = psf(psf_seq, pix_seq_g)
        pix_seq_r_blur = psf(psf_seq, pix_seq_r)
        for j, i in enumerate(seq):
            img_blur[i[1]][i[0]] = pix_seq_b_blur[j], pix_seq_g_blur[j], pix_seq_r_blur[j]
    img_blur = cv.medianBlur(img_blur, ksize=5)
    return img_blur


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


if __name__ == '__main__':
    img = cv.imread("origin.jpg")
    img_rotational_blur = rotation_blur(img=img, theta=0.1)
    cv.imwrite("origin_blur.jpg", img_rotational_blur)
