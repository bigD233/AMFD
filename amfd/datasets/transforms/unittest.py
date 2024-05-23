from mmdet.registry import TRANSFORMS
import warnings
from typing import Optional
from mmengine.registry import TRANSFORMS as MMCV_TRANSFORMS
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
import mmcv
from mmdet.datasets import CocoDataset
import cv2
import torch
from typing import List, Optional, Sequence, Tuple, Union
from numbers import Number
from mmengine.utils import is_seq_of
import math
from mmdet.registry import MODELS
import torch.nn as nn
import torch.nn.functional as F
from projects.Distillation.distillation.datasets.kaist_dataset import KAISTDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_image_with_bboxes(data_dict):
    # Load image using OpenCV
    image_1 = data_dict['img'][:,:,:3]/255
    image_2 = data_dict['img'][:,:,3:]/255
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image_1)
    ax2.imshow(image_2)
    image_path = data_dict['img_path']


    for bbox_info in data_dict['instances']:
        x_min, y_min, x_max, y_max = bbox_info['bbox']
        bbox_label = bbox_info['bbox_label']
        ignore_flag = bbox_info.get('ignore_flag', 0)  # 默认为0，如果不存在该键则返回0
        rect1 = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=1, edgecolor='r' if ignore_flag == 0 else 'g', facecolor='none'
        )

        rect2 = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=1, edgecolor='r' if ignore_flag == 0 else 'g', facecolor='none'
        )

        # 添加矩形框到Axes对象
        ax1.add_patch(rect1)
        ax2.add_patch(rect2)
    
    # ax1.set_title('Image 1 with Bounding Boxes')
    ax1.axis('off')  # 关闭坐标轴

    # ax2.set_title('Image 2 with Bounding Boxes')
    ax2.axis('off')  # 关闭坐标轴

    plt.subplots_adjust(wspace=0)  # 调整两个子图之间的空间
    plt.show()

if __name__=='__main__':
    ann_test='/media/yons/1/yxx/grad_proj_data/KAIST/anno/test_anno/KAIST_test_RGB_annotation.json'
    ann_train='/media/yons/1/yxx/grad_proj_data/KAIST/anno/train_anno/KAIST_train_RGB_annotation.json'
    dataset=KAISTDataset(ann_file=ann_train)
    li=TRANSFORMS.build(dict(type='LoadBGR3TFromKAIST'))
    for img in dataset:
        res=li.transform(img)
        visualize_image_with_bboxes(res)
    # 可选：打印数据集信息
    print(dataset)
