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
        return (image, mask)

    def __len__(self):
        return len(self.dataset)
    

class ExperimentalDatasetTest(Dataset):
    def __init__(self, root, transforms=None):
        self.transform = transforms
        self.files_defect = glob.glob(os.path.join(root) + "/*.*")
        
    def __getitem__(self, index):
        image = cv2.imread((self.files_defect[index]), cv2.IMREAD_GRAYSCALE)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        
        area = (self.files_defect[index][44:])
        area = int(area[0])
        
        if self.transform != None:
            image = self.transform(image)
            
        return image, area

    def __len__(self):
        return len(self.files_defect)


class ExperimentalDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.transform = transforms
        self.dataset = np.load(root[0])
        self.diameters = np.load(root[1])

    def __getitem__(self, index):
        image = torch.from_numpy(self.dataset[index])
        diameter = (self.diameters[index])

        image = image.unsqueeze(0)
        
        if self.transform != None:
            image = self.transform(image)
            
        return image, diameter

    def __len__(self):
        return len(self.dataset)

