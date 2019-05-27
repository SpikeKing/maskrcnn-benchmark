#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/5/21
"""
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import torch
import random
from maskrcnn_benchmark.utils import cv2_util

from matplotlib.collections import PatchCollection

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from demo.predictor import COCODemo
from maskrcnn_benchmark.config import cfg
from preprocess.csv2coco_processor import get_cate_dict
from preprocess.img_mask_drawer import get_center_of_polygon
from preprocess.pycococreatortools import binary_mask_to_polygon
from project_utils import mkdir_if_not_exist, traverse_dir_files, get_current_time_str, sort_two_list
from root_dir import ROOT_DIR


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
    def remove_overlay_core(n_mask_list):
        h, w = n_mask_list[0].shape
        img_origin = np.zeros(h * w, dtype=np.uint8)

        c_list = []
        c = 1
        for mask in reversed(n_mask_list):  # 逆序，先画概率低的，再画概率高的
            mask = np.reshape(mask, -1)  # 转成一行
            img_origin[mask == 1] = c
            c_list.append(c)
            c += 1

        mask_list = []
        for num in reversed(c_list):
            img_tmp = np.zeros(h * w, dtype=np.uint8)
            img_tmp[img_origin == num] = 1
            img_tmp_r = np.reshape(img_tmp, (h, w))
            mask_list.append(img_tmp_r)

        return mask_list

    @staticmethod
    def remove_overlay(masks, is_resize=True):
        mask_list = FashionPredictor.remove_overlay_core(masks)

        if is_resize:  # resize
            tmp_list = []
            for r_mask in mask_list:
                img_tmp_r = cv2.resize(r_mask, (512, 512), cv2.INTER_NEAREST)
                tmp_list.append(img_tmp_r)
            mask_list = FashionPredictor.remove_overlay_core(tmp_list)

        return mask_list

    # @staticmethod
    # def remove_overlay(masks):
    #     """
    #     已排序，(2, 1, 400, 226)
    #     """
    #     mask_list = FashionPredictor.remove_overlay(masks)
    #
    #     mask_np = np.array(mask_list)
    #     mask_np = np.expand_dims(mask_np, axis=1)
    #     print('[Info] mask_np: {}'.format(mask_np.shape))
    #
    #     mask_torch = torch.from_numpy(mask_np).float()
    #     print('[Info] mask_torch: {}'.format(mask_torch.shape))
    #     return mask_torch

    @staticmethod
    def create_blank(height, width, bgra_color):
        """Create new image(numpy array) filled with certain color in RGB"""
        # Create black blank image
        image = np.zeros((height, width, 4), np.uint8)

        # Since OpenCV uses BGR, convert the color first
        # Fill image with color
        image[:] = tuple([i * 255 for i in bgra_color])

        return image

    def show_mask_v2(self, img, predictions, img_name):
        """
        显示分割效果
        :param img: numpy的图像
        :param predictions: 分割结果
        :return: 显示分割效果
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, d = img.shape
        pylab.rcParams['figure.figsize'] = (8.0, 10.0)  # 图片尺寸
        plt.imshow(img)  # 需要提前填充图像

        extra_fields = predictions.extra_fields
        masks = extra_fields['mask']
        labels = extra_fields['labels']

        id_cat_dict = get_cate_dict()
        name_list = [id_cat_dict[int(l) - 1] for l in labels]

        np.random.seed(37)
        color_name_dict = dict()
        color_img_name_dict = dict()
        for name in name_list:
            c = (np.random.random((1, 3)) * 0.8 + 0.2).squeeze().tolist()
            c = tuple(c + [1.0])
            color_name_dict[name] = c

            img_colors = self.create_blank(h, w, c)
            color_img_name_dict[name] = img_colors

        for mask, name in zip(masks, name_list):
            c = color_name_dict[name]
            img_colors = color_img_name_dict[name]

            # 0~26是大类别, 其余是小类别 同时 每个标签只绘制一次
            tc = c - np.array([0.4, 0.4, 0.4, 0])  # 降低颜色
            tc = np.maximum(tc, 0.0)  # 最小值0

            bc = c - np.array([0.2, 0.2, 0.2, 0])  # 降低颜色
            bc = np.maximum(bc, 0.0)  # 最小值0

            mask = torch.squeeze(mask).data.numpy()

            img_mask = cv2.bitwise_and(img_colors, img_colors, mask=mask)

            contours, hierarchy = cv2_util.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(img_mask, contours, 0, bc * 255, 3)

            plt.imshow(img_mask, alpha=0.8)

            c_len = len(contours)

            random.shuffle(contours)
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    c_x = round(M['m10'] / M['m00'])
                    c_y = round(M['m01'] / M['m00'])

                    plt.text(c_x, c_y, name, ha='left', wrap=True, color=tc, fontsize=6,
                             bbox=dict(facecolor='white', alpha=0.5))  # 绘制标签
                    break

        ax = plt.gca()
        ax.set_autoscale_on(False)

        plt.axis('off')
        ax.get_xaxis().set_visible(False)  # this removes the ticks and numbers for x axis
        ax.get_yaxis().set_visible(False)  # this removes the ticks and numbers for y axis

        # plt.show()
        out_folder = os.path.join(ROOT_DIR, 'update', 'out-imgs')
        mkdir_if_not_exist(out_folder)
        out_file = os.path.join(out_folder, 'test.{}.png'.format(img_name))
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0, dpi=200)

        plt.close()  # 避免所有图像绘制在一起
        print('[Info] {} over'.format(img_name))

    @staticmethod
    def show_mask(img, predictions, img_name):
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
        # print('[Info] masks {}'.format(masks.shape))
        # masks = FashionPredictor.remove_overlay(masks)

        # name_list = [coco_demo.CATEGORIES[l] for l in labels]
        id_cat_dict = get_cate_dict()
        name_list = [id_cat_dict[int(l) - 1] for l in labels]

        seg_list = []
        for mask, label in zip(masks, labels):
            mask = torch.squeeze(mask)
            # print('[Info] masks {}'.format(mask.shape))
            segmentation = binary_mask_to_polygon(mask, tolerance=4)
            if segmentation:
                print('[Info] 无法绘制: {}'.format(get_cate_dict()[int(label)]))
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

            plt.text(c_x, c_y, name, ha='left', wrap=True, color=tc, fontsize=5,
                     bbox=dict(facecolor='white', alpha=0.5))  # 绘制标签

            polygons.append(pylab.Polygon(poly))  # 绘制多边形
            color.append(c)

        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.8)  # 添加多边形
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)  # 添加多边形的框
        ax.add_collection(p)

        plt.axis('off')
        ax.get_xaxis().set_visible(False)  # this removes the ticks and numbers for x axis
        ax.get_yaxis().set_visible(False)  # this removes the ticks and numbers for y axis

        out_folder = os.path.join(ROOT_DIR, 'update', 'out-imgs')
        mkdir_if_not_exist(out_folder)
        out_file = os.path.join(out_folder, 'test.{}.png'.format(img_name))
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0, dpi=200)

        plt.close()  # 避免所有图像绘制在一起

    @staticmethod
    def analyze_prediction(masks, labels):
        """
        调整masks和labels的格式
        """
        masks_list = []
        for mask in masks:
            mask_ep = FashionPredictor.decode_mask(mask)
            masks_list.append(mask_ep)

        labels_list = []
        for label in labels:
            labels_list.append(str(int(label) - 1))  # 第1个为背景

        return labels_list, masks_list

    def sorted_label_with_mask(self, labels_list, masks_list):
        idxes = sorted(range(len(labels_list)), key=lambda k: labels_list[k], reverse=True)
        r_labels = [labels_list[i] for i in idxes]
        r_masks = [masks_list[i] for i in idxes]

        return r_labels, r_masks

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        # used to make colors for each class
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = labels[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        o_img = image.copy()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, -1)

        added_image = cv2.addWeighted(o_img, 1.0, image, 0.5, 0)
        composite = added_image

        return composite

    def predict(self, img_path):
        img = cv2.imread(img_path)
        print('[Info] img size: {}'.format(img.shape))

        name = img_path.split('/')[-1]
        print('[Info] ImageId: {}'.format(name))

        predictions = self.coco_demo.compute_prediction(img)
        top_predictions = self.coco_demo.select_top_predictions(predictions)

        extra_fields = top_predictions.extra_fields
        masks_torch = extra_fields['mask']
        labels_list = extra_fields['labels'].tolist()

        i, _, _, _ = masks_torch.shape
        if i == 0:
            return [name], [0], ['1 1']

        masks_numpy = masks_torch.data.numpy()
        masks_list = []
        for mask in masks_numpy:
            n_mask = np.squeeze(mask, axis=0)
            masks_list.append(n_mask)

        labels, masks = self.sorted_label_with_mask(labels_list, masks_list)

        masks = FashionPredictor.remove_overlay(masks, is_resize=True)  # 是否Resize512
        labels_list, masks_list = self.analyze_prediction(masks, labels)

        # 显示去重之后的
        masks_torch = torch.stack(tuple([torch.from_numpy(i) for i in masks]))
        n, w, h = masks_torch.shape
        masks_torch = torch.reshape(masks_torch, (n, 1, w, h))

        extra_fields['labels'] = torch.from_numpy(np.array(labels))
        extra_fields['mask'] = masks_torch

        # self.show_mask_v2(img, top_predictions, name)

        csv_img, csv_ep, csv_label = [], [], []
        for label, mask in zip(labels_list, masks_list):
            csv_img.append(name)
            csv_label.append(label)
            csv_ep.append(mask)

        return csv_img, csv_label, csv_ep


def main():
    # test_folder = os.path.join(ROOT_DIR, 'datasets', 'test_mini5')
    test_folder = os.path.join(ROOT_DIR, 'datasets', 'test')
    # test_folder = '/data_sharing/data41_data1/zl9/fashion-2019/test'
    print('[Info] 数据集: {}'.format(test_folder))
    paths_list, names_list = traverse_dir_files(test_folder)
    # img_path = os.path.join(DATA_DIR, 'aoa-mina.jpeg')
    print('[Info] 图片总数: {}'.format(len(paths_list)))

    fp = FashionPredictor()

    csv_img_a, csv_label_a, csv_ep_a = [], [], []
    count = 0
    s_time = time.time()
    for name, path in zip(names_list, paths_list):
        img_path = path
        csv_img, csv_label, csv_ep = fp.predict(img_path)
        csv_img_a += csv_img
        csv_label_a += csv_label
        csv_ep_a += csv_ep

        count += 1
        print('[Info] count: {}'.format(count))
        # count += 1
        # if count == 5:
        #     break
    print('[Info] 耗时: {}'.format(time.time() - s_time))

    df = pd.DataFrame(
        {'ImageId': csv_img_a, 'EncodedPixels': csv_ep_a, 'ClassId': csv_label_a})
    df = df[['ImageId', 'EncodedPixels', 'ClassId']]  # change the column index

    csv_folder = os.path.join(ROOT_DIR, 'update', 'out')
    mkdir_if_not_exist(csv_folder)
    csv_file_name = os.path.join(csv_folder, 'fashion_2019.{}.csv'.format(get_current_time_str()))
    df.to_csv(csv_file_name, index=False, sep=str(','))


if __name__ == '__main__':
    main()
