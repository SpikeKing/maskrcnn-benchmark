#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/5/16
"""
import os
import numpy as np
import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib import pyplot as plt

from root_dir import ROOT_DIR
from project_utils import *


def load_json():
    val_file = os.path.join(ROOT_DIR, 'datasets', 'annotations', 'instances_val2017.json')
    data_line = read_file_utf8(val_file)[0]
    coco_dict = json.loads(data_line)
    print('Keys: {}'.format(coco_dict.keys()))

    info = coco_dict['info']
    licenses = coco_dict['licenses']
    images = coco_dict['images']
    annotations = coco_dict['annotations']
    categories = coco_dict['categories']

    print('-' * 50)
    print('[Info] info: {}'.format(info))  # 信息
    print('-' * 50)
    print('[Info] licenses: {}'.format(licenses))  # 8个licenses
    print('-' * 50)
    print('[Info] 图片数: {}'.format(len(images)))  # 图片数
    print('[Info] 图片: {}'.format(images[0]))  # 图片数
    print('-' * 50)
    print('[Info] 标注数: {}'.format(len(annotations)))  # 标注
    print('[Info] 标注: {}'.format(annotations[0]))  # 标注
    print('-' * 50)
    print('[Info] 类别数: {}'.format(len(categories)))  # 类别
    print('[Info] 类别: {}'.format(categories[0]))  # 类别

    return images, annotations


def draw_polygon(seg):
    print('[Info] 数据格式: {}'.format(seg))
    gemfield_polygons = seg
    polygons = []
    fig, ax = plt.subplots()

    gemfield_polygon = gemfield_polygons[0]
    max_value = max(gemfield_polygon) * 1.3
    gemfield_polygon = [i * 1.0 / max_value for i in gemfield_polygon]
    poly = np.array(gemfield_polygon).reshape((int(len(gemfield_polygon) / 2), 2))
    polygons.append(Polygon(poly, True))  # 多边形
    p = PatchCollection(polygons, cmap=matplotlib.cm.jet, alpha=0.4)
    colors = 100 * np.random.rand(1)
    p.set_array(np.array(colors))

    ax.add_collection(p)
    plt.show()


def draw_rle(seg):
    print('[Info] 数据格式: {}'.format(seg))
    rle = seg['counts']
    h, w = seg['size']
    M = np.zeros(h * w)
    N = len(rle)
    n = 0
    val = 1
    for pos in range(N):
        val = not val
        num = rle[pos]
        for c in range(num):
            M[n] = val
            n += 1
    gemfield = M.reshape(([h, w]), order='F')
    plt.imshow(gemfield)
    plt.show()


def process_img():
    images, annotations = load_json()

    id_name_dict = dict()
    for img in images:
        img_id = img['id']
        name = img['file_name']
        id_name_dict[img_id] = name

    count = 0
    for anno in annotations:

        iscrowd = anno['iscrowd']
        category_id = anno['category_id']
        # if iscrowd == 0 or category_id == 1:
        # if iscrowd == 0:
        #     continue
        print(anno)
        img_id = anno['image_id']
        img_name = id_name_dict[img_id]
        print('[Info] 图像名称: {}'.format(img_name))
        seg = anno['segmentation']

        count += 1
        if count == 2:
            draw_polygon(seg)
            break

    # print(images[0])


def main():
    # load_json()
    process_img()


if __name__ == '__main__':
    main()
