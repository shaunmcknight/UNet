# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:44:49 2022

@author: Shaun McKnight
"""

# import the necessary packages
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import os
import numpy as np
import cv2
import math

class SegmentationDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.transform = transforms
        
        #combine dataset to weight all defect sizes equally
        dataset = np.load(root)
        subset = dataset[0:24]
        self.dataset = np.concatenate((subset, dataset), 0)

        
    def __getitem__(self, index):
        image = torch.from_numpy(self.dataset[index][0])
        mask = torch.from_numpy(self.dataset[index][1])

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        if self.transform != None:
            image = self.transform(image)
            mask = self.transform(mask)
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
        image = image.unsqueeze(0)/image.max()
                
        file_name = os.path.basename(self.files_defect[index])      
        diameter = int(file_name[0])
        
        if self.transform != None:
            image = self.transform(image)

            
        return image, diameter

    def __len__(self):
        return len(self.files_defect)

class ExperimentalDatasetTVG(Dataset):
    def __init__(self, transforms=None, masked = False):
        root = r'C:\Users\Shaun McKnight\OneDrive - University of Strathclyde\PhD\Data\segmentation\test\TVG dataset + pad1-2-3\Shaun'
        masks_root = 'C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/PhD/Data/segmentation/test/TVG dataset + pad1-2-3/pad1.npy'
        
        self.masked = masked
        self.masks = np.load(masks_root)

        self.transform = transforms
        self.files_defect = glob.glob(os.path.join(root) + "/*.*")
        
    def __getitem__(self, index):
        image = cv2.imread((self.files_defect[index]), cv2.IMREAD_GRAYSCALE)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)/image.max()
                
        file_name = os.path.basename(self.files_defect[index])      
        diameter = int(file_name[0])
        
        if self.masked == True:
            image = image*self.masks[index]
            
        if self.transform != None:
            image = self.transform(image)

            
        return image, diameter

    def __len__(self):
        return len(self.files_defect)
    
    
class ExperimentalDatasetTest(Dataset):
    def __init__(self, root, transforms=None):
        self.transform = transforms
        self.files_defect = glob.glob(os.path.join(root) + "/*.*")
        self.diameters = np.load(r'C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/PhD/Data/segmentation/Experimental/ID018_test_dataset_maps.npy')

    def __getitem__(self, index):
        image = cv2.imread((self.files_defect[index]), cv2.IMREAD_GRAYSCALE)
        image = torch.from_numpy(image)
        # image = image.unsqueeze(0)/image.max()

        h, w = image.size()[0], image.size()[1]
        pad = (math.ceil(32-w/2), math.floor(32-w/2), math.ceil(32-h/2), math.floor(32-h/2))
        image = F.pad(image, pad, "constant", 0)     

        diameter = (self.diameters[index])
        
        if self.transform != None:
            image = self.transform(image)
            
        return image, diameter

    def __len__(self):
        return len(self.files_defect)


class ExperimentalDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.transform = transforms
        self.dataset = np.load(root[0])
        self.diameters = np.load(root[1])
        
        print('Dataset size ~ ', self.dataset.shape)

    def __getitem__(self, index):
        image = torch.from_numpy(self.dataset[index])
        diameter = (self.diameters[index])

        image = image.unsqueeze(0)
        
        if self.transform != None:
            image = self.transform(image)
            
        return image, diameter

    def __len__(self):
        return len(self.dataset)
    
    
class ExperimentalDatasetYoloMasked(Dataset):
    def __init__(self, root, transforms=None):
        self.transform = transforms
        self.dataset = np.load(root[0])
        self.diameters = np.load(root[1])
        self.masks = np.load(r'C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/PhD/Data/segmentation/test/yolo_masks/pad5.npy')

    def __getitem__(self, index):
        image = torch.from_numpy(self.dataset[index])
        diameter = (self.diameters[index])

        image = image.unsqueeze(0)
        masked_image = image*self.masks[index]#np.pad(self.masks[index], 96, mode='constant', constant_values=0)
        
        if self.transform != None:
            masked_image = self.transform(masked_image)
            
        return masked_image, diameter

    def __len__(self):
        return len(self.dataset)

