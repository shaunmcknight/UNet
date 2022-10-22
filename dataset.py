# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:44:49 2022

@author: Shaun McKnight
"""

# import the necessary packages
import torch 
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import os
import numpy as np
import cv2

class SegmentationDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.transform = transforms
        self.dataset = np.load(root)
        
    def __getitem__(self, index):
        image = torch.from_numpy(self.dataset[index][0])
        mask = torch.from_numpy(self.dataset[index][1])

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        if self.transform != None:
            image = self.transform(image)
        # print('Image shape ', np.shape(image))
        return (image, mask)

    def __len__(self):
        return len(self.dataset)
    

class ExperimentalDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.transform = transforms
        self.files_defect = np.load(glob.glob(os.path.join(root, "exp_defect") + "/*.*")[0])
        self.areas = "To be loaded"

        print('Max ', np.amax(self.files_defect))
        print('Min ', np.amin(self.files_defect))

    def __getitem__(self, index):
        image = torch.from_numpy(self.files_defect[index])
        image = image.unsqueeze(0)
        
        area = self.areas # to be updated
        
        if self.transform != None:
            image = self.transform(image)
            
        return image, area

    def __len__(self):
        return len(self.files_defect)

