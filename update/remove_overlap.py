#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/5/21

直接过滤CSV的Mask文件
"""

import pandas as pd
import os
import numpy as np
from tqdm import tqdm

from root_dir import ROOT_DIR


def get_mask(img_id, df, shape=(512, 512)):
    print(img_id)
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    px = df.loc[img_id]
    img_masks = px["EncodedPixels"]
    img_label = px["ClassId"]
    if type(img_masks) == float:
        return None
    elif type(img_masks) == str:
        img_masks = [img_masks]

    if type(img_label) == np.int64: img_label = [img_label]
    count = 1
    label_count = {}
    print(len(img_masks), type(img_label))
    for mask, label in zip(img_masks, img_label):
        if type(mask) == float:
            if len(px) == 1:
                return None
            else:
                continue

        if mask == "['1 1']":
            mask = '1 1'
        s = mask.split()

        label_count[count] = label
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            # keep previous prediction for overlapping pixels
            img[start:start + length] = count * (img[start:start + length] == 0)

        count += 1
    return img.reshape(shape).T, label_count


def decode_mask(mask, shape=(512, 512)):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if (len(runs) == 0): return np.nan
    runs[runs > shape[0] * shape[1]] = shape[0] * shape[1]
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def set_masks(mask, label_count):
    n = mask.max()
    result = []
    result_label = []
    for i in range(1, n + 1):
        result.append(decode_mask(mask == i))
        result_label.append(label_count[i])
    return result, result_label


def main():
    INPUT = os.path.join(ROOT_DIR, 'update', 'out', 'fashion_2019.20190523125113.csv')
    OUTPUT = os.path.join(ROOT_DIR, 'update', 'out', 'fashion_2019.20190523125113.submit.csv')

    pred_df = pd.read_csv(INPUT).set_index('ImageId')
    pred_df.head()

    names = list(set(pred_df.index))
    ship_list_dict = []
    for name in tqdm(names):
        mask, label_count = get_mask(name, pred_df)
        if (not isinstance(mask, np.ndarray) and mask == None) \
                or mask.sum() == 0:  # or name in test_names_nothing:
            ship_list_dict.append({'ImageId': name, 'EncodedPixels': np.nan, 'ClassId': np.nan})
        else:
            encodings, result_label = set_masks(mask, label_count)
            if len(encodings) == 0:
                ship_list_dict.append({'ImageId': name, 'EncodedPixels': np.nan, 'ClassId': np.nan})
                continue

            buf = []
            buf_l = []
            for e, l in zip(encodings, result_label):
                if e == e:
                    buf.append(e)
                    buf_l.append(l)
            encodings = buf
            if len(encodings) == 0: encodings = [np.nan]
            for encoding, label in zip(encodings, buf_l):
                ship_list_dict.append({'ImageId': name, 'EncodedPixels': encoding, 'ClassId': label})

    pred_df_cor = pd.DataFrame(ship_list_dict)
    pred_df_cor.to_csv(OUTPUT, index=False)


if __name__ == '__main__':
    main()
