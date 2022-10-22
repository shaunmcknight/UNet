# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:35:43 2022

@author: Shaun McKnight
"""
from dataset import SegmentationDataset
from torchvision import transforms

import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import math


def gen_diameter(area):
    pi = math.pi
    return 2*np.sqrt(area/pi)

def gen_db_mask(img):
    db_image = 20*np.log10(img/torch.max(img))
    db_mask = db_image
    
    db_mask[db_mask >= -6] = 1
    db_mask[db_mask < -6] = 0
    
    return db_mask

def calculate_area(mask):
    return np.count_nonzero(mask)/(0.8*0.8*4**2)


def prepare_plot(origImage, origMask, predMask):
    
    dbMask = gen_db_mask(origImage.squeeze())
    
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 10))
    
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage.squeeze())
    ax[1].imshow(origMask.squeeze())
    ax[2].imshow(predMask.squeeze())
    ax[3].imshow(dbMask.squeeze())
    
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask\n({:.1f} mm^2 ~ {:.1f} mm)".
                    format(calculate_area(origMask.squeeze()), 
                           gen_diameter(calculate_area(origMask.squeeze()))))
    ax[2].set_title("Predicted Mask\n({:.1f} mm^2 ~ {:.1f} mm)".
                    format(calculate_area(predMask.squeeze()), 
                           gen_diameter(calculate_area(predMask.squeeze()))))
    ax[3].set_title("6dB Mask\n({:.1f} mm^2 ~ {:.1f} mm)".
                    format(calculate_area(dbMask.squeeze()), 
                           gen_diameter(calculate_area(dbMask.squeeze()))))
    
    ax[0].grid(False)
    ax[1].grid(False)
    ax[2].grid(False)
    ax[3].grid(False)

    
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()
    
def eval_test(model):
    
	# set model to evaluation mode
    model.eval()
	
    transforms_ = transforms.Compose([transforms.ToPILImage(),
     	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
    		config.INPUT_IMAGE_WIDTH)),
         transforms.ToTensor()])
    
    # turn off gradient tracking
    with torch.no_grad():
        
        dataset = SegmentationDataset(root=config.DATASET_PATH, transforms=transforms_)
        for i in range(len(dataset)):
            image = dataset[i][0].to(config.DEVICE, dtype=torch.float)
            mask = dataset[i][1].to(config.DEVICE, dtype=torch.float)
            pred_mask = model(image.unsqueeze(0))
            pred_mask= torch.sigmoid(pred_mask)
            
            # filter out the weak predictions and convert them to integers
            pred_mask = (pred_mask > config.THRESHOLD) #* 255
            
                # pred_mask = (pred_mask-torch.min(pred_mask))
                # pred_mask = pred_mask/torch.max(pred_mask)
            
            # plt.figure()
            # plt.imshow(pred_mask.cpu().detach().squeeze())
            # plt.colorbar()
            # plt.clim([0.3, 1])
            # plt.show()
                                       
            # prepare a plot for visualization
            prepare_plot(image.cpu().detach(), mask.cpu().detach(), pred_mask.cpu().detach())
        

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

# make predictions and visualize the results
eval_test(unet)

#TODO:
    # plot defect sizings