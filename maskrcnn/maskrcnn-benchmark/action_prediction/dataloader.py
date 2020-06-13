# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:16:36 2019

@author: epyir
"""
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import os.path as osp
from PIL import Image
import json
import random

class BatchLoader(Dataset):
    def __init__(self, imageRoot, gtRoot, batchSize=1, cropSize=(1280, 720)):
        super(BatchLoader, self).__init__()
        
        self.imageRoot = imageRoot
        self.gtRoot = gtRoot
        self.cropSize = cropSize
        self.batchsize = batchSize
        
        with open(gtRoot) as json_file:
            data = json.load(json_file)
        
        # get image names and labels
        action_annotations = data['annotations']
        imgNames = data['images']
        self.imgNames, self.targets = [], []
        forward_sample = 0
        for i, img in enumerate(imgNames):
            if not (int(action_annotations[i]['category_id']) == 0 and forward_sample > 1000):
                self.imgNames.append(osp.join(self.imageRoot, img['file_name']))
                self.targets.append(int(action_annotations[i]['category_id']))
                forward_sample += 1
        
        
        self.count = len(self.imgNames)
        self.perm = list(range(self.count))
        random.shuffle(self.perm)
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, ind):
        imgName = self.imgNames[self.perm[ind]]
        target = np.array(self.targets[self.perm[ind]], dtype=np.int64)
        # target = one_hot(target, 4)
        
        img = np.array(Image.open(imgName))
        img = np.transpose(img, (2, 0, 1))
        img = torch.Tensor(img)
        batchDict = {
                'img': img,
                'target': target
                }
        return batchDict

def toDivisibility(img_list):
    # img_list is Tensor [BatchSize, 3, Height, Width]
    batchsize = img_list.shape[0]

    pass


# def make_one_hot(labels, C=4):
#     '''
#     Converts an integer label torch.autograd.Variable to a one-hot Variable.
#
#     Parameters
#     ----------
#     labels : torch.autograd.Variable of torch.cuda.LongTensor
#         N x 1 x H x W, where N is batch size.
#         Each value is an integer representing correct classification.
#     C : integer.
#         number of classes in labels.
#
#     Returns
#     -------
#     target : torch.autograd.Variable of torch.cuda.FloatTensor
#         N x C x H x W, where C is class number. One-hot encoded.
#     '''
#     one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
#     target = one_hot.scatter_(1, labels.data, 1)
#
#     target = Variable(target)
#
#     return target

# def one_hot(label, C=4):
#     one_hot_t = torch.LongTensor(1, C) % 3
#     one_hot_t = one_hot_t.zero_()
#     if label == 0:
#         one_hot_t[0, 0] = 1
#     elif label == 1:
#         one_hot_t[0, 1] = 1
#     elif label == 2:
#         one_hot_t[0, 2] = 1
#     elif label == 3:
#         one_hot_t[0, 3] = 1
#
#     return one_hot_t

