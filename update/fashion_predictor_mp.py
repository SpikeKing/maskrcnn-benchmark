#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/5/23
"""
import os
import sys
import time
import pandas as pd
import multiprocessing
from glob import glob

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from project_utils import mkdir_if_not_exist, get_current_time_str
from root_dir import ROOT_DIR
from update.fashion_predictor import FashionPredictor

NUM_WORKER = 4
MLP_GLOBAL = None


def build_model():
    global MLP_GLOBAL
    my_id = multiprocessing.current_process().pid if NUM_WORKER > 1 else 0
    MLP_GLOBAL = FashionPredictor()  # 多标签
    print("[Info] devs: {}".format(my_id))
    return MLP_GLOBAL


def detect_img(img_path):
    try:
        img_path = img_path
        csv_img, csv_label, csv_ep = MLP_GLOBAL.predict(img_path)
        return csv_img, csv_label, csv_ep
    except Exception as e:
        print('[Exception] img: {}, {}'.format(img_path, e))
        return [], [], []


def test_of_detect_img():
    img_path = "/Users/wang/workspace/maskrcnn-benchmark/datasets/test_mini5/0a4aae5ecd970a120bfcc6b377b6e187.jpg"
    build_model()
    csv_img, csv_label, csv_ep = detect_img(img_path)
    print(csv_img)


def process_imgs():
    s_time = time.time()

    # test_folder = '/Users/wang/workspace/maskrcnn-benchmark/datasets/test/'
    test_folder = '/data_sharing/data41_data1/zl9/fashion-2019/test/'  # 3
    # test_folder = '/data_sharing/data411/zl9/fashion-2019/test/'  # 2
    image_paths = glob(test_folder + '*.*')  # 全部图片
    print('[Info] 处理图片数: {}'.format(len(image_paths)))

    pool = multiprocessing.Pool(NUM_WORKER, initializer=build_model)
    res = pool.map_async(detect_img, image_paths, chunksize=1)

    while not res.ready():
        print("[Info] 待处理个数: {}".format(res._number_left))  # 获取
        time.sleep(1)  # 循环1秒打印

    result = res.get()

    pool.close()
    pool.join()

    imgs_list = [x[0] for x in result]
    labels_list = [x[1] for x in result]
    eps_list = [x[2] for x in result]

    # print(imgs_list)

    csv_img_a, csv_label_a, csv_ep_a = [], [], []
    for imgs, labels, eps in zip(imgs_list, labels_list, eps_list):
        csv_img_a += imgs
        csv_label_a += labels
        csv_ep_a += eps

    df = pd.DataFrame(
        {'ImageId': csv_img_a, 'EncodedPixels': csv_ep_a, 'ClassId': csv_label_a})
    df = df[['ImageId', 'EncodedPixels', 'ClassId']]  # change the column index

    csv_folder = os.path.join(ROOT_DIR, 'update', 'out')
    mkdir_if_not_exist(csv_folder)
    csv_file_name = os.path.join(csv_folder, 'fashion_2019.{}.csv'.format(get_current_time_str()))
    df.to_csv(csv_file_name, index=False, sep=str(','))

    print('[Info] 耗时: {}'.format(time.time() - s_time))


def main():
    # test_of_detect_img()
    process_imgs()


if __name__ == '__main__':
    main()
