# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:35:43 2022

@author: Shaun McKnight
"""

import torch
import os

CURRENT_DIR = os.getcwd()

# base path of the dataset
DATASET_PATH = ('C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/PhD/Data/segmentation/a_scan_noise_w_sementation_masks.npy')
TEST_DATASET_PATHS = ('C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/PhD/Data/segmentation/Experimental/ID018_test_dataset.npy', 
                     'C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/PhD/Data/segmentation/Experimental/ID018_test_dataset_maps.npy')
# define the test split
TEST_SPLIT = 0.10

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 500 # 250 #100
BATCH_SIZE = 32

# define the input image dimensions
INPUT_IMAGE_WIDTH = 256 #128
INPUT_IMAGE_HEIGHT = 256 #128

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(CURRENT_DIR, BASE_OUTPUT, "unet_ut.pth")
PLOT_PATH = os.path.join(CURRENT_DIR, BASE_OUTPUT, "plot.png")
TEST_PATHS = os.path.join(CURRENT_DIR, BASE_OUTPUT, "test_paths.txt")