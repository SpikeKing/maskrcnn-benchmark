#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/5/8
"""
import os
import time

import cv2

import pylab
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

from demo.predictor import COCODemo
from maskrcnn_benchmark.config import cfg
from preprocess.csv2coco_processor import get_cate_dict
from preprocess.img_mask_drawer import get_center_of_polygon
from preprocess.pycococreatortools import binary_mask_to_polygon
from project_utils import mkdir_if_not_exist
from root_dir import DATA_DIR, ROOT_DIR


def show_cv_img(img_cv):
    img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_path = os.path.join(DATA_DIR, 'test.jpg')
    cv2.imwrite(img_path, img_cv)

    plt.imshow(img)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    pylab.show()


def remove_overlay(masks):
    """
    已排序，(2, 1, 400, 226)
    """
    masks = masks.data.numpy()
    print('[Info] masks: {}'.format(masks.shape))
    _, _, h, w = masks.shape
    img_origin = np.zeros(h * w, dtype=np.uint8)

    c_list = []
    c = 1
    for mask in reversed(masks):  # 逆序，先画概率低的，再画概率高的
        mask = np.reshape(mask, -1)
        img_origin[mask == 1] = c
        c_list.append(c)
        c += 1
        # mask_list.append(mask)

    mask_list = []
    for num in reversed(c_list):
        img_tmp = np.zeros(h * w, dtype=np.uint8)
        img_tmp[img_origin == num] = 1

        img_tmp = np.reshape(img_tmp, (h, w))
        mask_list.append(img_tmp)

    mask_np = np.array(mask_list)
    mask_np = np.expand_dims(mask_np, axis=1)
    print('[Info] mask_np: {}'.format(mask_np.shape))

    mask_torch = torch.from_numpy(mask_np).float()
    print('[Info] mask_torch: {}'.format(mask_torch.shape))
    return mask_torch


def show_mask(img, predictions, coco_demo):
    """
    显示分割效果
    :param img: numpy的图像
    :param predictions: 分割结果
    :param coco_demo: 函数集
    :return: 显示分割效果
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)  # 图片尺寸
    plt.imshow(img)  # 需要提前填充图像
    # plt.show()

    extra_fields = predictions.extra_fields
    masks = extra_fields['mask']
    labels = extra_fields['labels']
    print('[Info] masks {}'.format(masks.shape))
    masks = remove_overlay(masks)

    # name_list = [coco_demo.CATEGORIES[l] for l in labels]
    id_cat_dict = get_cate_dict()
    name_list = [id_cat_dict[int(l) - 1] for l in labels]

    seg_list = []
    for mask in masks:
        mask = torch.squeeze(mask)
        print('[Info] masks {}'.format(mask.shape))
        segmentation = binary_mask_to_polygon(mask, tolerance=0)
        if segmentation:
            seg_list.append(segmentation[0])

    ax = plt.gca()
    ax.set_autoscale_on(False)

    polygons, color = [], []

    np.random.seed(37)

    for name, seg in zip(name_list, seg_list):
        c = (np.random.random((1, 3)) * 0.8 + 0.2).tolist()[0]

        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
        c_x, c_y = get_center_of_polygon(poly)  # 计算多边形的中心点

        # 0~26是大类别, 其余是小类别 同时 每个标签只绘制一次
        tc = c - np.array([0.5, 0.5, 0.5])  # 降低颜色
        tc = np.maximum(tc, 0.0)  # 最小值0

        plt.text(c_x, c_y, name, ha='left', wrap=True, color=tc,
                 bbox=dict(facecolor='white', alpha=0.5))  # 绘制标签

        polygons.append(pylab.Polygon(poly))  # 绘制多边形
        color.append(c)

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)  # 添加多边形
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)  # 添加多边形的框
    ax.add_collection(p)

    plt.axis('off')
    ax.get_xaxis().set_visible(False)  # this removes the ticks and numbers for x axis
    ax.get_yaxis().set_visible(False)  # this removes the ticks and numbers for y axis

    out_folder = os.path.join(ROOT_DIR, 'demo', 'out')
    mkdir_if_not_exist(out_folder)
    out_file = os.path.join(out_folder, 'test.png'.format())
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0, dpi=200)

    plt.close()  # 避免所有图像绘制在一起


def main():
    img_path = os.path.join(DATA_DIR, 'aoa-mina.jpeg')

    img = cv2.imread(img_path)
    print('[Info] img size: {}'.format(img.shape))
    # show_cv_img(img)

    # config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
    config_file = "../configs/e2e_mask_rcnn_R_50_FPN_1x.my.yaml"

    cfg.merge_from_file(config_file)  # 设置配置文件
    cfg.merge_from_list(["MODEL.MASK_ON", True])
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])  # 指定为CPU

    coco_demo = COCODemo(  # 创建模型文件
        cfg,
        # show_mask_heatmaps=True,
        min_image_size=800,
        confidence_threshold=0.7,
    )

    s_time = time.time()
    for i in range(1000):
        predictions = coco_demo.compute_prediction(img)
    e_time = time.time()
    print('[Info] Time: {}'.format(e_time-s_time))
    top_predictions = coco_demo.select_top_predictions(predictions)
    show_mask(img, top_predictions, coco_demo)

    print('执行完成!')


if __name__ == '__main__':
    main()
