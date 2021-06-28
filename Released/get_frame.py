# -*- coding: utf-8 -*-

import cv2
import os
import json, random
from PIL import Image
import numpy as np


def process(src, des):
    videos = os.listdir(src)
    for video_name in videos:
        file_name = video_name.split('.')[0]
        print(file_name)

        vid_cap = cv2.VideoCapture(src+video_name)
        count = 0
        # videos = list()
        success, image = vid_cap.read()
        while success:
            temp = vid_cap.get(0)
            # print(temp)
            count += 1
            cv2.imwrite(des + file_name + '_' + str(count) + ".png", image)
            clc = 0.2 * 1000 * count
            vid_cap.set(cv2.CAP_PROP_POS_MSEC, clc)
            success, image = vid_cap.read()

            if count >= 41:
                break

            if temp == vid_cap.get(0):
                print("视频异常，结束循环")
                break


if __name__ == '__main__':
    src = 'KoNViD_1k_videos/'
    des = 'frames/'
    process(src, des)







