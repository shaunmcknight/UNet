# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:35:43 2022

@author: Shaun McKnight
"""

from dataset import SegmentationDataset

from model import *
from config import *
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import numpy as np

def sampleDataImage(dataloader):
    test_data = next(iter(dataloader))
    i = 1
    plt.figure()
    plt.subplot(1,2,1)
    plt.gca().set_title('UT image')
    plt.gca().grid(False)
    plt.imshow(test_data[0][i].squeeze())
    plt.subplot(1,2,2)
    plt.gca().set_title('Mask')
    plt.gca().grid(False)
    plt.imshow(test_data[1][i].squeeze())
    plt.show()
    
# define transformations
transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])

# create the train and test datasets

completeDS = SegmentationDataset(root=config.DATASET_PATH, transforms=transforms)
    
split = 0.7

split_train, split_test = (round(len(completeDS)*split), 
                           len(completeDS)-round(len(completeDS)*split))

trainDS, testDS = torch.utils.data.random_split(
    completeDS, (split_train, split_test))

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the testing set...")

# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)
# 	num_workers=os.cpu_count())

testLoader = DataLoader(testDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)
# 	num_workers=os.cpu_count())

sampleDataImage(trainLoader)

# initialize our UNet model
unet = UNet().to(config.DEVICE)

# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)

# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	# set the model in training mode
	unet.train()
    
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0
    
	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
        
		# perform a forward pass and calculate the training loss
		pred = unet(x)
		loss = lossFunc(pred, y)
        
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
        
		# add the loss to the total training loss so far
		totalTrainLoss += loss
        
	# switch off autograd
    
	with torch.no_grad():
		# set the model in evaluation mode
		unet.eval()
        
		# loop over the validation set
		for (x, y) in testLoader:
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            
			# make the predictions and calculate the validation loss
			pred = unet(x)
			totalTestLoss += lossFunc(pred, y)
            
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps
    
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
    
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))
    
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig(config.PLOT_PATH)
plt.show()

# serialize the model to disk
torch.save(unet, config.MODEL_PATH)

