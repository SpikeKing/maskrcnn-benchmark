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
import matplotlib.pyplot as plt

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

from root_dir import DATA_DIR


def show_cv_img(img_cv):
    img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_path = os.path.join(DATA_DIR, 'test.jpg')
    cv2.imwrite(img_path, img_cv)

    plt.imshow(img)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    pylab.show()


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

    s = time.time()
    # for i in range(20):
    predictions = coco_demo.run_on_opencv_image(img)
    print('[Info] {}'.format(time.time() - s))
    print('执行完成!')
    show_cv_img(predictions)


if __name__ == '__main__':
    main()
