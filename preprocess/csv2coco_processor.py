#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/5/17
"""
import os
import sys
import json
import re
import fnmatch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime
from skimage.data import imread

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from preprocess import pycococreatortools
from root_dir import ROOT_DIR

INFO = {
    "year": 2019,
    "version": "1.0",
    "description": "The 2019 FGVC^6 iMaterialist Competition - Fashion track dataset.",
    "contributor": "iMaterialist Fashion Competition group",
    "url": "https://github.com/visipedia/imat_comp",
    "date_created": datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        "id": 0,
        "name": "shirt, blouse",  # 衬衫
        "supercategory": "upperbody",
        "level": 2
    },
    {
        "id": 1,
        "name": "top, t-shirt, sweatshirt",  # T恤衫
        "supercategory": "upperbody",
        "level": 2
    },
    {
        "id": 2,
        "name": "sweater",  # 毛衣
        "supercategory": "upperbody",
        "level": 2
    },
    {
        "id": 3,
        "name": "cardigan",  # 羊毛衫
        "supercategory": "upperbody",
        "level": 2
    },
    {
        "id": 4,
        "name": "jacket",  # 夹克
        "supercategory": "upperbody",
        "level": 2
    },
    {
        "id": 5,
        "name": "vest",  # 背心
        "supercategory": "upperbody",
        "level": 2
    },
    {
        "id": 6,
        "name": "pants",  # 裤子
        "supercategory": "lowerbody",
        "level": 2
    },
    {
        "id": 7,
        "name": "shorts",  # 短裤
        "supercategory": "lowerbody",
        "level": 2
    },
    {
        "id": 8,
        "name": "skirt",  # 短裙
        "supercategory": "lowerbody",
        "level": 2
    },
    {
        "id": 9,
        "name": "coat",  # 外套
        "supercategory": "wholebody",
        "level": 2
    },
    {
        "id": 10,
        "name": "dress",  # 连衣裙
        "supercategory": "wholebody",
        "level": 2
    },
    {
        "id": 11,
        "name": "jumpsuit",  # 连身衣裤
        "supercategory": "wholebody",
        "level": 2
    },
    {
        "id": 12,
        "name": "cape",  # 披肩
        "supercategory": "wholebody",
        "level": 2
    },
    {
        "id": 13,
        "name": "glasses",  # 眼镜
        "supercategory": "head",
        "level": 2
    },
    {
        "id": 14,
        "name": "hat",  # 帽子
        "supercategory": "head",
        "level": 2
    },
    {
        "id": 15,
        "name": "headband, head covering, hair accessory",  # 发饰
        "supercategory": "head",
        "level": 2
    },
    {
        "id": 16,
        "name": "tie",  # 领带
        "supercategory": "neck",
        "level": 2
    },
    {
        "id": 17,
        "name": "glove",  # 手套
        "supercategory": "arms and hands",
        "level": 2
    },
    {
        "id": 18,
        "name": "watch",  # 手表
        "supercategory": "arms and hands",
        "level": 2
    },
    {
        "id": 19,
        "name": "belt",  # 腰带
        "supercategory": "waist",
        "level": 2
    },
    {
        "id": 20,
        "name": "leg warmerleg warmer",  # 护腿
        "supercategory": "legs and feet",
        "level": 2
    },
    {
        "id": 21,
        "name": "tights, stockings",  # 长袜
        "supercategory": "legs and feet",
        "level": 2
    },
    {
        "id": 22,
        "name": "sock",  # 短袜
        "supercategory": "legs and feet",
        "level": 2
    },
    {
        "id": 23,
        "name": "shoe",  # 鞋
        "supercategory": "legs and feet",
        "level": 2
    },
    {
        "id": 24,
        "name": "bag, wallet",  # 包
        "supercategory": "others",
        "level": 2
    },
    {
        "id": 25,
        "name": "scarf",  # 围巾
        "supercategory": "others",
        "level": 2
    },
    {
        "id": 26,
        "name": "umbrella",  # 雨伞
        "supercategory": "others",
        "level": 2
    },
    {
        "id": 27,
        "name": "hood",  # 兜帽
        "supercategory": "garment parts",
        "level": 2
    },
    {
        "id": 28,
        "name": "collar",  # 衣领
        "supercategory": "garment parts",
        "level": 2
    },
    {
        "id": 29,
        "name": "lapel",  # 翻领
        "supercategory": "garment parts",
        "level": 2
    },
    {
        "id": 30,
        "name": "epaulette",  # 简章
        "supercategory": "garment parts",
        "level": 2
    },
    {
        "id": 31,
        "name": "sleeve",
        "supercategory": "garment parts",
        "level": 2
    },
    {
        "id": 32,
        "name": "pocket",
        "supercategory": "garment parts",
        "level": 2
    },
    {
        "id": 33,
        "name": "neckline",
        "supercategory": "garment parts",
        "level": 2
    },
    {
        "id": 34,
        "name": "buckle",
        "supercategory": "closures",
        "level": 2
    },
    {
        "id": 35,
        "name": "zipper",
        "supercategory": "closures",
        "level": 2
    },
    {
        "id": 36,
        "name": "applique",
        "supercategory": "decorations",
        "level": 2
    },
    {
        "id": 37,
        "name": "bead",
        "supercategory": "decorations",
        "level": 2
    },
    {
        "id": 38,
        "name": "bow",
        "supercategory": "decorations",
        "level": 2
    },
    {
        "id": 39,
        "name": "flower",
        "supercategory": "decorations",
        "level": 2
    },
    {
        "id": 40,
        "name": "fringe",
        "supercategory": "decorations",
        "level": 2
    },
    {
        "id": 41,
        "name": "ribbon",
        "supercategory": "decorations",
        "level": 2
    },
    {
        "id": 42,
        "name": "rivet",
        "supercategory": "decorations",
        "level": 2
    },
    {
        "id": 43,
        "name": "ruffle",
        "supercategory": "decorations",
        "level": 2
    },
    {
        "id": 44,
        "name": "sequin",
        "supercategory": "decorations",
        "level": 2
    },
    {
        "id": 45,
        "name": "tassel",
        "supercategory": "decorations",
        "level": 2
    }

]  # 来源于数据集中的label_descriptions.json文件


def get_cate_dict():
    id_name_dict = dict()
    for sub_dict in CATEGORIES:
        cid = sub_dict['id']
        name = sub_dict['name']
        id_name_dict[cid] = name
    return id_name_dict


# 本地
# IMAGE_DIR = os.path.join(ROOT_DIR, 'datasets', 'train_minimal')
IMAGE_DIR = os.path.join(ROOT_DIR, 'datasets', 'test_mini5')
# csv_file = os.path.join(ROOT_DIR, 'datasets', 'train.csv')
csv_file = os.path.join(ROOT_DIR, 'datasets', 'fashion_2019.20190521143725.csv')


# 线上
# IMAGE_DIR = "/data_sharing/data41_data1/zl9/fashion-2019/train"
# csv_file = "/data_sharing/data41_data1/zl9/fashion-2019/train.csv"


def get_current_time_str():
    """
    输入当天的日期格式, 20170718_1137
    :return: 20170718_1137
    """
    return datetime.now().strftime('%Y%m%d%H%M%S')


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# mask_rle(string) --> rle_decode() -->  np.ndarry(np.unit8)
# shape: (height,width) , 1 - mask, 0 - background
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
    return img.reshape(shape).T  # Needed to align to RLE direction


def filter_for_jpeg(root, files):
    """
    过滤图片
    """
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files


def save_bad_ann(image_name, mask, segmentation_id, class_id):
    """
    存储标注信息
    """
    img = imread(os.path.join(IMAGE_DIR, image_name))

    fig, axarr = plt.subplots(1, 3)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(mask)
    axarr[2].imshow(img)
    axarr[2].imshow(mask, alpha=0.4)
    fig.set_size_inches(20, 16)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)

    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    plt.savefig(
        os.path.join('./tmp', image_name.split('.')[0] + '_' + str(class_id) + '_' + str(segmentation_id) + '.png'))
    plt.close()


def process_csv_to_coco():
    """
    将
    """

    # df = pd.read_csv(csv_file, nrows=10)  # 读取CSV文件, 前10行
    # for index, row in df.iterrows():
    #     print(row)
    df = pd.read_csv(csv_file)  # 读取CSV文件
    print("[Info] Dataframe lines: {}, items: {}".format(df.shape[0], df.shape[1]))  # 行数和列数

    df = df.dropna(axis=0)  # 处理坏行
    n_data = df.shape[0]
    print("[Info] Instances:", n_data)

    # 最终放进json文件里的字典
    coco_output = {  # COCO格式
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],  # 放一个空列表占位置，后面再append
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # 最外层的循环是图片，因为图片的基本信息需要共享
    # IMAGE_DIR路径下找到所有的图片
    n_item = 0
    for root, _, files in os.walk(IMAGE_DIR):
        image_paths = filter_for_jpeg(root, files)  # 图片文件地址
        num_of_image_files = len(image_paths)  # 图片个数

        # 遍历每一张图片
        for image_path in image_paths:
            # 提取图片信息
            try:
                image = Image.open(image_path)
            except Exception as e:
                print('[Exception] 图片异常: {} {}'.format(image_path, e))
                continue

            image_name = os.path.basename(image_path)  # 不需要具体的路径，只要图片文件名
            image_info = pycococreatortools.create_image_info(image_id, image_name, image.size)
            coco_output["images"].append(image_info)  # 创建图片信息

            # 内层循环是mask，把每一张图片的mask搜索出来
            # print(df.loc[df['ImageId'] == image_name])
            rle_masks = df.loc[df['ImageId'] == image_name, 'EncodedPixels'].tolist()
            rle_masks_label = df.loc[df['ImageId'] == image_name, 'ClassId'].tolist()
            num_of_rle_masks = len(rle_masks)

            for index in range(num_of_rle_masks):  # 处理多个Mask信息
                binary_mask = rle_decode(rle_masks[index], image.size)  # 转换标注格式
                class_id = rle_masks_label[index]  # 所有图片的类别，可能包含类别和属性信息

                category_info = {'id': class_id, 'is_crowd': 0}  # 0是多边形格式
                annotation_info = pycococreatortools.create_annotation_info(  # 转换为多边形格式
                    segmentation_id, image_id, category_info, binary_mask,
                    image.size, tolerance=2)

                # 不是所有的标注都会被转换,低质量标注会被过滤掉
                # 正常的标注加入数据集，不好的标注保存供观察
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                    # else:
                # save_bad_ann(image_name, binary_mask, segmentation_id, class_id)

                # 无论标注是否被写入数据集，均分配一个编号
                segmentation_id = segmentation_id + 1

            print("%d of %d is done." % (image_id, num_of_image_files))
            image_id = image_id + 1

            n_item += 1
            if n_item == 40:
                break

    # 存储JSON数据
    out_file_name = 'instances_train2019.fashion.{}.{}.json'.format(n_item, get_current_time_str())
    out_file = os.path.join(ROOT_DIR, 'datasets', out_file_name)
    print('[Info] 存储文件: {}'.format(out_file))
    with open(out_file, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

    print('[Info] COCO数据集生成结束! ')


def main():
    process_csv_to_coco()


if __name__ == '__main__':
    main()
