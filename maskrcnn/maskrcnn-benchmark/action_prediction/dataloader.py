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


