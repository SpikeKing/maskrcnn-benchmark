#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/5/21
"""

import os
import json

from root_dir import ROOT_DIR
from project_utils import read_file_utf8, get_current_time_str


def clean_json():
    # json_file = os.path.join(ROOT_DIR, 'datasets', 'instances_train2019.fashion.20190521030953.json')
    json_file = os.path.join(ROOT_DIR, 'datasets', 'instances_train2019.fashion.clean.20190521162739.json')
    json_dict = json.loads(read_file_utf8(json_file)[0])
    print(json_dict.keys())

    coco_output = {  # COCO格式
        "info": json_dict['info'],
        "licenses": json_dict['licenses'],
        "categories": json_dict['categories'],
        "images": json_dict['images'],  # 放一个空列表占位置，后面再append
        "annotations": []
    }

    annotations = json_dict['annotations']
    for anno in annotations:
        category_id = anno['category_id']
        str_list = category_id.split('_')
        if len(str_list) > 1:
            print(anno)
            break
        n_category_id = str_list[0]
        anno['category_id'] = n_category_id

    coco_output["annotations"] = annotations

    out_file_name = 'instances_train2019.fashion.clean.{}.json'.format(get_current_time_str())
    out_file = os.path.join(ROOT_DIR, 'datasets', out_file_name)
    print('[Info] 存储文件: {}'.format(out_file))
    with open(out_file, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

    print('[Info] 清理完成!')


def main():
    clean_json()


if __name__ == '__main__':
    main()
