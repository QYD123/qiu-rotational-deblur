# -*- coding: UTF-8 -*-
# @Time    : 22/04/2019 17:43
# @Author  : QYD
import torch
import numpy as np
import functools
from models.fpn_inception import FPNInception
import torch.nn as nn
import cv2 as cv

model_use = ["FPNInception", "FPNMobileNet"]


def post_process(x: torch.Tensor) -> np.ndarray:
    x, = x
    x = x.detach().cpu().float().numpy()
    x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
    return x.astype('uint8')


def blurred_field_cal(img, theta):
    h, w, c = img.shape
    center_x = h // 2
    center_y = w // 2
    yy, xx = np.meshgrid(np.arange(w), np.arange(h))  # 得到列坐标和行坐标
    blurred_field = np.sqrt(
        (xx - center_x) ** 2 + (yy - center_y) ** 2) * theta  # 根据行坐标和列坐标计算模糊域.可以把计算得到的模糊域给存下来不需要重复的计算。

    blurred_field = np.reshape(blurred_field, (h, w, 1))
    return blurred_field.astype(np.int8)


def normal_img(img, mean=0.5, std=0.5):
    img = ((img / 255) - mean) / std
    return img


def q_deblurGAN(img, theta, use_gpu=False, weights="./weights/fpn_inception.h5"):
    blurred_field = blurred_field_cal(img, theta)
    img_input = np.concatenate([img, blurred_field], axis=2)
    img_input = normal_img(img_input, mean=0.5, std=0.5)
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = torch.tensor(img_input, dtype=torch.float32)
    img_input = img_input.unsqueeze(dim=0)
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    model = FPNInception(norm_layer=norm_layer)
    model.train()
    if use_gpu:
        img_input = img_input.cuda()
        model.cuda()
    model.load_state_dict({k.replace(
        'module.',
        ''): v
                           for k, v
                           in torch.load(weights)["model"].items()})
    out = model(img_input)
    out = post_process(out)
    return out


if __name__ == '__main__':
    img = cv.imread("../samples/origin_blur.jpg")
    img_deblur = q_deblurGAN(img=img, theta=0.05, use_gpu=True)
    cv.imwrite("../samples/origin_blurGAN.jpg", img_deblur)
