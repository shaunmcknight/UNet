# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:35:43 2022

@author: Shaun McKnight
"""
from dataset import SegmentationDataset, ExperimentalDataset
from torchvision import transforms
from sklearn.metrics import mean_squared_error


import config
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
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

def scatter_area(results):

    index_3 = np.where(np.array(results["gt"]) == 7.421874999999998)
    index_6 = np.where(np.array(results["gt"]) == 27.734374999999993)
    index_9 = np.where(np.array(results["gt"]) == 63.281249999999986)
    
    fig = plt.figure(figsize = (10,10))
    plt.suptitle("Area segmentation results", size=30)
    
    plt.subplot(3,1,1)
    plt.scatter(np.array(results["pred"])[index_3], np.array(results["gt"])[index_3], label="Predicted", marker = "d", s = 150)
    plt.scatter(np.array(results["db"])[index_3], np.array(results["gt"])[index_3], label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.array(results["gt"])[index_3], np.array(results["gt"])[index_3], label="Ground truth", marker = "|", s=1000, c = 'g')
        
    plt.subplot(3,1,2)
    plt.scatter(np.array(results["pred"])[index_6], np.array(results["gt"])[index_6], label="Predicted", marker = "d", s = 150)
    plt.scatter(np.array(results["db"])[index_6], np.array(results["gt"])[index_6], label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.array(results["gt"])[index_6], np.array(results["gt"])[index_6], label="Ground truth", marker = "|", s=1000, c = 'g')
    
    plt.subplot(3,1,3)
    plt.scatter(np.array(results["pred"])[index_9], np.array(results["gt"])[index_9], label="Predicted", marker = "d", s = 150)
    plt.scatter(np.array(results["db"])[index_9], np.array(results["gt"])[index_9], label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.array(results["gt"])[index_9], np.array(results["gt"])[index_9], label="Ground truth", marker = "|", s=1000, c = 'g')
    plt.figlegend(labels = ['UNet prediction', '6 dB drop', 'Ground truth'], ncol = 3, borderpad=1.8, loc = 'lower center')
    
    fig.supylabel("Ground truth area (mm^2)", ha = 'left')
    plt.xlabel("Predicted area (mm^2)")
    # plt.savefig(config.PLOT_PATH)
    plt.show()
    
def scatter_diameter(results):

    index_3 = np.where(np.array(results["gt"]) == 7.421874999999998)
    index_6 = np.where(np.array(results["gt"]) == 27.734374999999993)
    index_9 = np.where(np.array(results["gt"]) == 63.281249999999986)
    
    fig = plt.figure(figsize = (10,10))
    plt.suptitle("Radius segmentation results", size=30)
    
    plt.subplot(3,1,1)
    plt.scatter(gen_diameter(np.array(results["pred"])[index_3]), np.round(gen_diameter(np.array(results["gt"])[index_3])), label="Predicted", marker = "d", s = 150)
    plt.scatter(gen_diameter(np.array(results["db"])[index_3]), np.round(gen_diameter(np.array(results["gt"])[index_3])), label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.round(gen_diameter(np.array(results["gt"])[index_3])), np.round(gen_diameter(np.array(results["gt"])[index_3])), label="Ground truth", marker = "|", s=1000, c = 'g')
           
    plt.subplot(3,1,2)
    plt.scatter(gen_diameter(np.array(results["pred"])[index_6]), np.round(gen_diameter(np.array(results["gt"])[index_6])), label="Predicted", marker = "d", s = 150)
    plt.scatter(gen_diameter(np.array(results["db"])[index_6]), np.round(gen_diameter(np.array(results["gt"])[index_6])), label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.round(gen_diameter(np.array(results["gt"])[index_6])), np.round(gen_diameter(np.array(results["gt"])[index_6])), label="Ground truth", marker = "|", s=1000, c = 'g')
           
    plt.subplot(3,1,3)
    plt.scatter(gen_diameter(np.array(results["pred"])[index_9]), np.round(gen_diameter(np.array(results["gt"])[index_9])), label="Predicted", marker = "d", s = 150)
    plt.scatter(gen_diameter(np.array(results["db"])[index_9]), np.round(gen_diameter(np.array(results["gt"])[index_9])), label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.round(gen_diameter(np.array(results["gt"])[index_9])), np.round(gen_diameter(np.array(results["gt"])[index_9])), label="Ground truth", marker = "|", s=1000, c = 'g')
      
    plt.figlegend(labels = ['UNet prediction', '6 dB drop', 'Ground truth'], ncol = 3, borderpad=1.8, loc = 'lower center')
     
    fig.supylabel("Ground truth area (mm)", ha = 'left')
    plt.xlabel("Predicted area (mm)")
    # plt.savefig(config.PLOT_PATH)
    plt.show()
    
def visualise_areas(results):
   
    index_3 = np.where(np.array(results["gt"]) == 7.421874999999998)
    index_6 = np.where(np.array(results["gt"]) == 27.734374999999993)
    index_9 = np.where(np.array(results["gt"]) == 63.281249999999986)
    
    average_3_pred = np.mean(np.array(results["pred"])[index_3])
    average_6_pred = np.mean(np.array(results["pred"])[index_6])
    average_9_pred = np.mean(np.array(results["pred"])[index_9])
        
    average_3_db = np.mean(np.array(results["db"])[index_3])
    average_6_db = np.mean(np.array(results["db"])[index_6])
    average_9_db = np.mean(np.array(results["db"])[index_9])
        
    circle3 = plt.Circle((5, 0), (3/2), color='g')
    circle3_pred = plt.Circle((5, 0), gen_diameter(average_3_pred)/2, color='blue')
    circle3_db = plt.Circle((5, 0), gen_diameter(average_3_db)/2, color='r')
        
    circle6 = plt.Circle((15, 0), (6/2), color='g')
    circle6_pred = plt.Circle((15, 0), gen_diameter(average_6_pred)/2, color='blue')
    circle6_db = plt.Circle((15, 0), gen_diameter(average_6_db)/2, color='r')
        
    circle9 = plt.Circle((30, 0), (9/2), color='g')
    circle9_pred = plt.Circle((30, 0), gen_diameter(average_9_pred)/2, color='blue')
    circle9_db = plt.Circle((30, 0), gen_diameter(average_9_db)/2, color='r')
    
    fig, ax = plt.subplots()#figsize = (10,10)) # note we must use plt.subplots, not plt.subplot
    # (or if you have an existing figure)
    # fig = plt.gcf()
    # ax = fig.gca()
    ax.set_aspect('equal')
    ax.set_xlim(0,50)
    ax.set_ylim(-10,10)
    ax.add_patch(circle3_db)
    ax.add_patch(circle3_pred)
    ax.add_patch(circle3)
    
    ax.add_patch(circle6_db)
    ax.add_patch(circle6_pred)
    ax.add_patch(circle6)
    
    ax.add_patch(circle9_db)
    ax.add_patch(circle9_pred)
    ax.add_patch(circle9)
    
    # ax.annotate(text = '3mm', xy=(5-1.5,0), xytext=(5+1.5,0),  arrowprops=dict(arrowstyle='<->'))
    # ax.annotate(text = '6mm', xy=(15-3,0), xytext=(15+3,0),  arrowprops=dict(arrowstyle='<->'))
    # ax.annotate(text = '9mm', xy=(30-4.5,0), xytext=(30+4.5,0),  arrowprops=dict(arrowstyle='<->'))

    ax.arrow(5-(3/2), 0, 3, 0, head_width=0.5, head_length=0.5, fc='k', ec='k', length_includes_head= True, shape='full')
    ax.arrow(5+(3/2), 0, -3, 0, head_width=0.5, head_length=0.5, fc='k', ec='k', length_includes_head= True, shape='full')
    
    ax.arrow(15-(3), 0, 6, 0, head_width=0.5, head_length=0.5, fc='k', ec='k', length_includes_head= True, shape='full')
    ax.arrow(15+(3), 0, -6, 0, head_width=0.5, head_length=0.5, fc='k', ec='k', length_includes_head= True, shape='full')
    
    ax.arrow(30-4.5, 0, 9, 0, head_width=0.5, head_length=0.5, fc='k', ec='k', length_includes_head= True, shape='full')
    ax.arrow(30+4.5, 0, -9, 0, head_width=0.5, head_length=0.5, fc='k', ec='k', length_includes_head= True, shape='full')

    ax.annotate('3mm', (5, 8), ha='center')
    ax.annotate('6mm', (15, 8), ha='center')
    ax.annotate('9mm', (30, 8), ha='center')
    
    plt.figlegend(labels = ['6 dB average size', 'UNet average size', 'Ground truth'], ncol = 3, loc = 'lower center')
    plt.suptitle("Comparison of average defect sizing")
    plt.tight_layout()
    plt.subplots_adjust(top=1)
  
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

def scatter_area_exp(results):

    index_3 = np.where(np.array(results["gt"]) == 7.0685834705770345)
    index_4 = np.where(np.array(results["gt"]) == 12.566370614359172)
    index_6 = np.where(np.array(results["gt"]) == 28.274333882308138)
    index_7 = np.where(np.array(results["gt"]) == 38.48451000647496)
    index_9 = np.where(np.array(results["gt"]) == 63.61725123519331)
    
    fig = plt.figure(figsize = (10,10))
    plt.suptitle("Area segmentation results", size=30)
    
    plt.subplot(5,1,1)
    plt.scatter(np.array(results["pred"])[index_3], np.array(results["gt"])[index_3], label="Predicted", marker = "d", s = 150)
    plt.scatter(np.array(results["db"])[index_3], np.array(results["gt"])[index_3], label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.array(results["gt"])[index_3], np.array(results["gt"])[index_3], label="Ground truth", marker = "|", s=1000, c = 'g')
        
    plt.subplot(5,1,2)
    plt.scatter(np.array(results["pred"])[index_4], np.array(results["gt"])[index_4], label="Predicted", marker = "d", s = 150)
    plt.scatter(np.array(results["db"])[index_4], np.array(results["gt"])[index_4], label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.array(results["gt"])[index_4], np.array(results["gt"])[index_4], label="Ground truth", marker = "|", s=1000, c = 'g')
            
    plt.subplot(5,1,3)
    plt.scatter(np.array(results["pred"])[index_6], np.array(results["gt"])[index_6], label="Predicted", marker = "d", s = 150)
    plt.scatter(np.array(results["db"])[index_6], np.array(results["gt"])[index_6], label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.array(results["gt"])[index_6], np.array(results["gt"])[index_6], label="Ground truth", marker = "|", s=1000, c = 'g')
    
    plt.subplot(5,1,4)
    plt.scatter(np.array(results["pred"])[index_7], np.array(results["gt"])[index_7], label="Predicted", marker = "d", s = 150)
    plt.scatter(np.array(results["db"])[index_7], np.array(results["gt"])[index_7], label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.array(results["gt"])[index_7], np.array(results["gt"])[index_7], label="Ground truth", marker = "|", s=1000, c = 'g')
    plt.figlegend(labels = ['UNet prediction', '6 dB drop', 'Ground truth'], ncol = 3, borderpad=1.8, loc = 'lower center')
    
    plt.subplot(5,1,5)
    plt.scatter(np.array(results["pred"])[index_9], np.array(results["gt"])[index_9], label="Predicted", marker = "d", s = 150)
    plt.scatter(np.array(results["db"])[index_9], np.array(results["gt"])[index_9], label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.array(results["gt"])[index_9], np.array(results["gt"])[index_9], label="Ground truth", marker = "|", s=1000, c = 'g')
    plt.figlegend(labels = ['UNet prediction', '6 dB drop', 'Ground truth'], ncol = 3, borderpad=1.8, loc = 'lower center')
    
    fig.supylabel("Ground truth area (mm^2)", ha = 'left')
    plt.xlabel("Predicted area (mm^2)")
    # plt.savefig(config.PLOT_PATH)
    plt.show()
    
def scatter_diameter_exp(results):

    index_3 = np.where(np.array(results["gt"]) == 7.0685834705770345)
    index_4 = np.where(np.array(results["gt"]) == 12.566370614359172)
    index_6 = np.where(np.array(results["gt"]) == 28.274333882308138)
    index_7 = np.where(np.array(results["gt"]) == 38.48451000647496)
    index_9 = np.where(np.array(results["gt"]) == 63.61725123519331)
    
    fig = plt.figure(figsize = (10,10))
    plt.suptitle("Radius segmentation results", size=30)
    
    plt.subplot(5,1,1)
    plt.scatter(gen_diameter(np.array(results["pred"])[index_3]), np.round(gen_diameter(np.array(results["gt"])[index_3])), label="Predicted", marker = "d", s = 150)
    plt.scatter(gen_diameter(np.array(results["db"])[index_3]), np.round(gen_diameter(np.array(results["gt"])[index_3])), label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.round(gen_diameter(np.array(results["gt"])[index_3])), np.round(gen_diameter(np.array(results["gt"])[index_3])), label="Ground truth", marker = "|", s=1000, c = 'g')
        
    plt.subplot(5,1,2)
    plt.scatter(gen_diameter(np.array(results["pred"])[index_4]), np.round(gen_diameter(np.array(results["gt"])[index_4])), label="Predicted", marker = "d", s = 150)
    plt.scatter(gen_diameter(np.array(results["db"])[index_4]), np.round(gen_diameter(np.array(results["gt"])[index_4])), label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.round(gen_diameter(np.array(results["gt"])[index_4])), np.round(gen_diameter(np.array(results["gt"])[index_4])), label="Ground truth", marker = "|", s=1000, c = 'g')
            
    plt.subplot(5,1,3)
    plt.scatter(gen_diameter(np.array(results["pred"])[index_6]), np.round(gen_diameter(np.array(results["gt"])[index_6])), label="Predicted", marker = "d", s = 150)
    plt.scatter(gen_diameter(np.array(results["db"])[index_6]), np.round(gen_diameter(np.array(results["gt"])[index_6])), label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.round(gen_diameter(np.array(results["gt"])[index_6])), np.round(gen_diameter(np.array(results["gt"])[index_6])), label="Ground truth", marker = "|", s=1000, c = 'g')
            
    plt.subplot(5,1,4)
    plt.scatter(gen_diameter(np.array(results["pred"])[index_7]), np.round(gen_diameter(np.array(results["gt"])[index_7])), label="Predicted", marker = "d", s = 150)
    plt.scatter(gen_diameter(np.array(results["db"])[index_7]), np.round(gen_diameter(np.array(results["gt"])[index_7])), label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.round(gen_diameter(np.array(results["gt"])[index_7])), np.round(gen_diameter(np.array(results["gt"])[index_7])), label="Ground truth", marker = "|", s=1000, c = 'g')
            
    plt.subplot(5,1,5)
    plt.scatter(gen_diameter(np.array(results["pred"])[index_9]), np.round(gen_diameter(np.array(results["gt"])[index_9])), label="Predicted", marker = "d", s = 150)
    plt.scatter(gen_diameter(np.array(results["db"])[index_9]), np.round(gen_diameter(np.array(results["gt"])[index_9])), label="6 dB sizing", marker = "d", s = 150)
    plt.scatter(np.round(gen_diameter(np.array(results["gt"])[index_9])), np.round(gen_diameter(np.array(results["gt"])[index_9])), label="Ground truth", marker = "|", s=1000, c = 'g')
        
    plt.figlegend(labels = ['UNet prediction', '6 dB drop', 'Ground truth'], ncol = 3, borderpad=1.8, loc = 'lower center')
     
    fig.supylabel("Ground truth area (mm)", ha = 'left')
    plt.xlabel("Predicted area (mm)")
    # plt.savefig(config.PLOT_PATH)
    plt.show()
      
def scatter_diameter_compare_exp(results):

    index_3 = np.where(np.array(results["gt"]) == 7.0685834705770345)
    index_4 = np.where(np.array(results["gt"]) == 12.566370614359172)
    index_6 = np.where(np.array(results["gt"]) == 28.274333882308138)
    index_7 = np.where(np.array(results["gt"]) == 38.48451000647496)
    index_9 = np.where(np.array(results["gt"]) == 63.61725123519331)
    
    average_3_pred = gen_diameter(np.mean(np.array(results["pred"])[index_3]))
    average_4_pred = gen_diameter(np.mean(np.array(results["pred"])[index_4]))
    average_6_pred = gen_diameter(np.mean(np.array(results["pred"])[index_6]))
    average_7_pred = gen_diameter(np.mean(np.array(results["pred"])[index_7]))
    average_9_pred = gen_diameter(np.mean(np.array(results["pred"])[index_9]))
        
    average_3_db = gen_diameter(np.mean(np.array(results["db"])[index_3]))
    average_4_db = gen_diameter(np.mean(np.array(results["db"])[index_4]))
    average_6_db = gen_diameter(np.mean(np.array(results["db"])[index_6]))
    average_7_db = gen_diameter(np.mean(np.array(results["db"])[index_7]))
    average_9_db = gen_diameter(np.mean(np.array(results["db"])[index_9]))
    
    gt = [3,4,6,7,9]
    pred = [average_3_pred, average_4_pred, average_6_pred, average_7_pred, average_9_pred]
    db = [average_3_db, average_4_db, average_6_db, average_7_db, average_9_db]    
    
    plt.figure(figsize = (10,10))
    plt.scatter(gt, pred, marker='o')
    plt.scatter(gt, db, marker='o')
    plt.xlim([0, 14])
    plt.ylim([0, 14])
    plt.gca().plot([0, 1], [0, 1], transform=plt.gca().transAxes, ls='--', color = 'g')
    plt.axline(xy1=(0, 0.5), slope=1, ls='--', color = 'grey')
    plt.axline(xy1=(0, -0.5), slope=1, ls='--', color = 'grey')

    plt.legend(('UNet prediction ~ RMSE {}'.format(round(mean_squared_error(gt, pred),3)),
                '6dB drop ~ RMSE {}'.format(round(mean_squared_error(gt, db),3)),
                'Ground truth',
                '+/- 0.5mm'))
    plt.xlabel('Ground truth defect radius (mm)')
    plt.ylabel('Predicted defect radius (mm)')
    plt.title("Experimental test segmentation results", size=30)
    # plt.savefig(config.PLOT_PATH)
    plt.show()
    
def visualise_areas_exp(results):
    def add_circles(center, radius_gt, area_pred, area_db):
        # fig, ax = plt.subplots()#figsize = (10,10)) # note we must use plt.subplots, not plt.subplot

        circle = plt.Circle(center, (radius_gt/2), facecolor='None', alpha=1, linewidth=1.5, linestyle='--', edgecolor='#D81B60')
        circle_pred = plt.Circle(center, gen_diameter(area_pred)/2, color='#1E88E5')
        circle_db = plt.Circle(center, gen_diameter(area_db)/2, color='#E0A800')
        
        patches = [circle_db, circle_pred, circle]
                
        return center, patches
    
    def add_dim(ax, center, diameter):
        center[0]
        ax.arrow(center[0]-(diameter/2), 0, diameter, 0, head_width=0.5, head_length=0.5, fc='k', ec='k', length_includes_head= True, shape='full')
        ax.arrow(center[0]+(diameter/2), 0, -diameter, 0, head_width=0.5, head_length=0.5, fc='k', ec='k', length_includes_head= True, shape='full')
        
        ax.annotate('{}mm'.format(diameter), (center[0], 8), ha='center')
   
    index_3 = np.where(np.array(results["gt"]) == 7.0685834705770345)
    index_4 = np.where(np.array(results["gt"]) == 12.566370614359172)
    index_6 = np.where(np.array(results["gt"]) == 28.274333882308138)
    index_7 = np.where(np.array(results["gt"]) == 38.48451000647496)
    index_9 = np.where(np.array(results["gt"]) == 63.61725123519331)
    
    average_3_pred = np.mean(np.array(results["pred"])[index_3])
    average_4_pred = np.mean(np.array(results["pred"])[index_4])
    average_6_pred = np.mean(np.array(results["pred"])[index_6])
    average_7_pred = np.mean(np.array(results["pred"])[index_7])
    average_9_pred = np.mean(np.array(results["pred"])[index_9])
        
    average_3_db = np.mean(np.array(results["db"])[index_3])
    average_4_db = np.mean(np.array(results["db"])[index_4])
    average_6_db = np.mean(np.array(results["db"])[index_6])
    average_7_db = np.mean(np.array(results["db"])[index_7])
    average_9_db = np.mean(np.array(results["db"])[index_9])
        
    total_patches = []
    offset = 3
    
    fig, ax = plt.subplots()#figsize = (10,10)) # note we must use plt.subplots, not plt.subplot

    center = [5,0]
    center, patches = add_circles(center, 3, average_3_pred, average_3_db)
    [ax.add_patch(patch) for patch in patches]
    add_dim(ax, center, 3)
 
    center = [center[0] + (offset + gen_diameter(average_3_db)/2 + gen_diameter(average_4_db)/2), 0]
    center, patches = add_circles(center, 4, average_4_pred, average_4_db)
    [ax.add_patch(patch) for patch in patches]
    add_dim(ax, center, 4)

    center = [center[0] + (offset + gen_diameter(average_4_db)/2 + gen_diameter(average_6_db)/2), 0]
    center, patches = add_circles(center, 6, average_6_pred, average_6_db)
    [ax.add_patch(patch) for patch in patches]
    add_dim(ax, center, 6)

    center = [center[0] + (offset + gen_diameter(average_6_db)/2 + gen_diameter(average_7_db)/2), 0]
    center, patches = add_circles(center, 7, average_7_pred, average_7_db)
    [ax.add_patch(patch) for patch in patches]
    add_dim(ax, center, 7)

    center = [center[0] + (offset + gen_diameter(average_7_db)/2 + gen_diameter(average_9_db)/2), 0]
    center, patches = add_circles(center, 9, average_9_pred, average_9_db)
    [ax.add_patch(patch) for patch in patches]
    add_dim(ax, center, 9)

    ax.set_aspect('equal')
    ax.set_xlim(0,60)
    ax.set_ylim(-10,10)
    
    plt.figlegend(labels = ['6 dB average size', 'UNet average size', 'Ground truth'], ncol = 3, loc = 'lower center')
    plt.suptitle("Comparison of average defect sizing")
    plt.tight_layout()
    plt.subplots_adjust(top=1)
    plt.show()
    
def eval_test(model, results):
    
	# set model to evaluation mode
    model.eval()
	
    transforms_ = transforms.Compose([transforms.ToPILImage(),
     	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
    		config.INPUT_IMAGE_WIDTH)), transforms.ToTensor()])
    
    # turn off gradient tracking
    with torch.no_grad():
        
        dataset = SegmentationDataset(root=config.DATASET_PATH, transforms=transforms_)
        for i in range(len(dataset)):
            image = dataset[i][0].to(config.DEVICE, dtype=torch.float)
            mask = dataset[i][1].to(config.DEVICE, dtype=torch.float)
            pred_mask = model(image.unsqueeze(0))
            pred_mask= torch.sigmoid(pred_mask)
            
            # filter out the weak predictions and convert them to integers
            # pred_mask = (pred_mask > config.THRESHOLD) #* 255
            pred_mask = np.round(pred_mask.cpu().detach(), decimals=1)
            
            #detch results
            image = image.cpu().detach()
            mask =  mask.cpu().detach()
            
            results["gt"].append(calculate_area(mask.squeeze()))
            results["pred"].append(calculate_area(pred_mask.squeeze()))
            results["db"].append(calculate_area(gen_db_mask(image.squeeze())))
            
            # plt.figure()
            # plt.imshow(pred_mask.cpu().detach().squeeze())
            # plt.colorbar()
            # # plt.clim([0.3, 1])
            # plt.show()
                              
            # print(pred_mask.cpu().detach().squeeze())
            # prepare a plot for visualization
            # prepare_plot(image, mask, pred_mask)
        
    return results

def eval_experimental(model, results):
    # set model to evaluation mode
    model.eval()
	
    transforms_ = transforms.Compose([transforms.ToPILImage(),
     	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
    		config.INPUT_IMAGE_WIDTH), interpolation=0), transforms.ToTensor()])
    
    # turn off gradient tracking
    with torch.no_grad():
        dataset = ExperimentalDataset(root=config.TEST_DATASET_PATHS, transforms=transforms_)
        for i in range(len(dataset)):
            image = dataset[i][0].to(config.DEVICE, dtype=torch.float)
            diameter = dataset[i][1]
            area = math.pi*(diameter/2)**2
            
            pred_mask = model(image.unsqueeze(0))
            pred_mask= torch.sigmoid(pred_mask)
            
            # filter out the weak predictions and convert them to integers
            # pred_mask = (pred_mask > config.THRESHOLD) #* 255
            # pred_mask = pred_mask.cpu().detach()
            pred_mask = np.round(pred_mask.cpu().detach(), decimals=1)
            pred_mask[pred_mask > 0] = 1
            #detch results
            image = image.cpu().detach()
            
            results["gt"].append(area)
            results["pred"].append(calculate_area(pred_mask.squeeze()))
            results["db"].append(calculate_area(gen_db_mask(image.squeeze())))
            
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(image.squeeze())
            plt.subplot(1,3,2)
            plt.imshow(pred_mask.cpu().detach().squeeze())
            plt.subplot(1,3,3)
            plt.imshow(gen_db_mask(image.squeeze()))
            # plt.clim([0.3, 1])
            plt.show()
                              
            # print(pred_mask.cpu().detach().squeeze())
            # prepare a plot for visualization
            
            print('GT Area {} ~ Predicted {}'.format(area, calculate_area(pred_mask.squeeze())))
            # prepare_plot(image, mask, pred_mask)
        
    return results

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
model_epoch = 180
unet = torch.load(config.model_path(model_epoch)).to(config.DEVICE)

#dict of results
results = {
    "gt":[],
    "pred":[],
    "db":[]
    }

# make predictions and visualize the results based on test data
results = eval_test(unet, results)
scatter_area(results)
scatter_diameter(results)
visualise_areas(results)

# make predictions based on experimental data
results = eval_experimental(unet, results)
scatter_diameter_exp(results)
scatter_diameter_compare_exp(results)
visualise_areas_exp(results)

#TODO:
    # plot defect sizings