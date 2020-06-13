# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:16:36 2019

@author: epyir
"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import os.path as osp
from PIL import Image
from scipy.ndimage import zoom
import json
import random

from maskrcnn_benchmark.data.transforms import transforms as T


class BatchLoader(Dataset):
    def __init__(self, imageRoot, gtRoot, reasonRoot=None, batchSize=1, cropSize=(1280, 720)):
        super(BatchLoader, self).__init__()

        self.imageRoot = imageRoot
        self.gtRoot = gtRoot
        self.cropSize = cropSize
        self.batchsize = batchSize

        with open(gtRoot) as json_file:
            data = json.load(json_file)
        with open(reasonRoot) as json_file:
            reason = json.load(json_file)
        data['images'] = sorted(data['images'], key=lambda k: k['file_name'])
        reason = sorted(reason, key=lambda k: k['file_name'])

        # get image names and labels
        action_annotations = data['annotations']
        imgNames = data['images']
        self.imgNames, self.targets, self.reasons = [], [], []
        for i, img in enumerate(imgNames):
            ind = img['id']
            # print(len(action_annotations[ind]['category']))
            if len(action_annotations[ind]['category']) == 4  or action_annotations[ind]['category'][4] == 0:
                file_name = osp.join(self.imageRoot, img['file_name'])
                if os.path.isfile(file_name):
                    self.imgNames.append(file_name)
                    self.targets.append(torch.LongTensor(action_annotations[ind]['category']))
                    self.reasons.append(torch.LongTensor(reason[i]['reason']))

        '''# get image names and labels
        action_annotations = data['annotations']
        imgNames = data['images']
        self.imgNames, self.targets = [], []
        for i, img in enumerate(imgNames):
            if action_annotations[i]['category'][4] == 0:
                self.imgNames.append(osp.join(self.imageRoot, img['file_name']))
                self.targets.append(torch.LongTensor(action_annotations[i]['category']))'''

        self.count = len(self.imgNames)
        print("number of samples in dataset:{}".format(self.count))
        self.perm = list(range(self.count))
        random.shuffle(self.perm)

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        # test = True
        imgName = self.imgNames[self.perm[ind]]
        target = np.array(self.targets[self.perm[ind]], dtype=np.int64)
        reason = np.array(self.reasons[self.perm[ind]], dtype=np.int64)
        # target = one_hot(target, 4)

        img_ = Image.open(imgName)
        # img_ = img_.resize((640, 320))

        color_jitter = T.ColorJitter(
                brightness=0.0,
                contrast=0.0,
                saturation=0.0,
                hue=0.0,
            )
        normalize_transform = T.Normalize(
                mean=[102.9801, 115.9465, 122.7717],
                std=[1., 1., 1.],
                to_bgr255=True,
            )
        transform = T.Compose(
                [color_jitter,
                 #T.Resize(self.cropSize[1], self.cropSize[0]),
                 T.ToTensor(),
                 normalize_transform,
                ]
            )
        img, target = transform(img_, target)
        batchDict = {
                'img': img,
                'target': torch.FloatTensor(target)[0:4],
                'ori_img': np.array(img_),
                'reason': torch.FloatTensor(reason)
            }
        return batchDict
