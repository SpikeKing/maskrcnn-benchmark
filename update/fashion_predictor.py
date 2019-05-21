#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/5/21
"""
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pylab
import torch

import pandas as pd

from matplotlib.collections import PatchCollection

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from demo.predictor import COCODemo
from maskrcnn_benchmark.config import cfg
from preprocess.csv2coco_processor import get_cate_dict
from preprocess.img_mask_drawer import get_center_of_polygon
from preprocess.pycococreatortools import binary_mask_to_polygon
from project_utils import mkdir_if_not_exist, traverse_dir_files, get_current_time_str
from root_dir import ROOT_DIR, DATA_DIR


class FashionPredictor(object):
    def __init__(self):
        config_file = os.path.join(ROOT_DIR, 'configs', 'e2e_mask_rcnn_R_50_FPN_1x.my.yaml')  # 配置文件

        cfg.merge_from_file(config_file)  # 设置配置文件
        cfg.merge_from_list(["MODEL.MASK_ON", True])
        # cfg.merge_from_list(["MODEL.DEVICE", "cpu"])  # 指定为CPU
        cfg.merge_from_list(["MODEL.DEVICE", "cuda"])  # 指定为GPU

        self.coco_demo = COCODemo(  # 创建模型文件
            cfg,
            min_image_size=800,
            confidence_threshold=0.7,
        )

    @staticmethod
    def decode_mask(mask):
        pixels = mask.T.flatten()
        # pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        if len(runs) == 0:
            return np.nan
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    @staticmethod
    def rle_decode(mask_rle, shape=(768, 768)):
        """
        将Mask的格式由点-宽格式，转换为RLE格式
        """
        s = mask_rle.split()
        starts = np.asarray(s[0::2], dtype=int)  # 开始
        lengths = np.asarray(s[1::2], dtype=int)  # 终止

        # starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)  # 图片大小
        # img = np.full(shape[0] * shape[1], 0, dtype=np.uint8)
        for lo, hi in zip(starts, ends):  # 起始和终止
            img[lo:hi] = 1
        # return img.reshape(shape).T  # Needed to align to RLE direction
        return img.reshape(shape)  # Needed to align to RLE direction

    @staticmethod
    def generate_masks_list(masks):
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

            # TODO: 提交要求
            resized_binary_mask = cv2.resize(img_tmp, (512, 512), cv2.INTER_NEAREST)

            mask_list.append(resized_binary_mask)

            # 测试
            # mask_ep = FashionPredictor.decode_mask(img_tmp)
            # img_tmp2 = FashionPredictor.rle_decode(mask_ep, img_tmp.shape)
            # mask_list.append(img_tmp2)

        return mask_list

    @staticmethod
    def remove_overlay(masks):
        """
        已排序，(2, 1, 400, 226)
        """
        mask_list = FashionPredictor.generate_masks_list(masks)

        mask_np = np.array(mask_list)
        mask_np = np.expand_dims(mask_np, axis=1)
        print('[Info] mask_np: {}'.format(mask_np.shape))

        mask_torch = torch.from_numpy(mask_np).float()
        print('[Info] mask_torch: {}'.format(mask_torch.shape))
        return mask_torch

    @staticmethod
    def show_mask(img, predictions):
        """
        显示分割效果
        :param img: numpy的图像
        :param predictions: 分割结果
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
        masks = FashionPredictor.remove_overlay(masks)

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

        out_folder = os.path.join(ROOT_DIR, 'update', 'out')
        mkdir_if_not_exist(out_folder)
        out_file = os.path.join(out_folder, 'test.png'.format())
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0, dpi=200)

        plt.close()  # 避免所有图像绘制在一起

    @staticmethod
    def analyze_prediction(predictions):
        extra_fields = predictions.extra_fields
        masks = extra_fields['mask']
        labels = extra_fields['labels']

        masks = FashionPredictor.generate_masks_list(masks)

        masks_list = []
        for mask in masks:
            mask_ep = FashionPredictor.decode_mask(mask)
            masks_list.append(mask_ep)

        labels_list = []
        for label in labels:
            labels_list.append(str(int(label) - 1))  # 第1个为背景

        return labels_list, masks_list

    def predict(self, img_path):
        img = cv2.imread(img_path)
        print('[Info] img size: {}'.format(img.shape))

        name = img_path.split('/')[-1]
        print('[Info] ImageId: {}'.format(name))

        predictions = self.coco_demo.compute_prediction(img)
        top_predictions = self.coco_demo.select_top_predictions(predictions)

        # self.show_mask(img, top_predictions)
        labels_list, masks_list = self.analyze_prediction(top_predictions)
        # print(labels_list)
        # print(masks_list)

        if not labels_list:
            return [name], [0], [["1 1"]],

        csv_img, csv_ep, csv_label = [], [], []
        for label, mask in zip(labels_list, masks_list):
            csv_img.append(name)
            csv_label.append(label)
            csv_ep.append(mask)

        return csv_img, csv_label, csv_ep


def main():
    # test_folder = os.path.join(ROOT_DIR, 'datasets', 'test_mini5')
    test_folder = '/data_sharing/data41_data1/zl9/fashion-2019/test'
    paths_list, names_list = traverse_dir_files(test_folder)
    # img_path = os.path.join(DATA_DIR, 'aoa-mina.jpeg')

    fp = FashionPredictor()

    csv_img_a, csv_label_a, csv_ep_a = [], [], []
    count = 0
    for name, path in zip(names_list, paths_list):
        img_path = path
        csv_img, csv_label, csv_ep = fp.predict(img_path)
        csv_img_a += csv_img
        csv_label_a += csv_label
        csv_ep_a += csv_ep
        print('[Info] count: {}'.format(count))
        # count += 1
        # if count == 5:
        #     break

    df = pd.DataFrame(
        {'ImageId': csv_img_a, 'EncodedPixels': csv_ep_a, 'ClassId': csv_label_a})
    df = df[['ImageId', 'EncodedPixels', 'ClassId']]  # change the column index

    csv_file_name = os.path.join(ROOT_DIR, 'update', 'out', 'fashion_2019.{}.csv'.format(get_current_time_str()))
    df.to_csv(csv_file_name, index=False, sep=str(','))


if __name__ == '__main__':
    main()
