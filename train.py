# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:35:43 2022

@author: Shaun McKnight
"""

from dataset import SegmentationDataset, ExperimentalDataset
from sklearn.metrics import mean_absolute_error

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
import math
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
# transforms_ = transforms.Compose([transforms.ToPILImage(),
#  	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
# 		config.INPUT_IMAGE_WIDTH), interpolation=0), 
#         transforms.RandomAffine(degrees = 0, scale = (0.8, 1.2)),
# 	transforms.ToTensor()])

transforms_ = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH), interpolation=0),
	transforms.ToTensor()])

transforms_exp_ = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH), interpolation=0),
     transforms.ToTensor()])

# create the train and test datasets

completeDS = SegmentationDataset(root=config.DATASET_PATH, transforms=transforms_)

experimentalDS = ExperimentalDataset(root=config.TEST_DATASET_PATHS, transforms=transforms_exp_)
    
split = 0.7

split_train, split_test = (round(len(completeDS)*split), 
                           len(completeDS)-round(len(completeDS)*split))

trainDS, testDS = torch.utils.data.random_split(
    completeDS, (split_train, split_test))

split = 0
split_train, split_test = (round(len(experimentalDS)*split), 
                           len(experimentalDS)-round(len(experimentalDS)*split))
exp_validDS, exp_testDS = torch.utils.data.random_split(
    experimentalDS, (split_train, split_test))

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the testing set...")

# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)
# 	num_workers=os.cpu_count())

testLoader = DataLoader(testDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)
# 	num_workers=os.cpu_count())

exp_testLoader = DataLoader(experimentalDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)

sampleDataImage(trainLoader)

# initialize our UNet model
unet = UNet().to(config.DEVICE)

# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)

# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE
exp_testSteps = len(exp_testDS) // config.BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": [], "test_loss_exp": []}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	# set the model in training mode
    unet.train()
    
	# initialize the total training and validation loss
    totalTrainLoss = 0
    totalTestLoss = 0
    totalTestLossExp = 0
    
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
            
        test_results = {"gt": [], "pred": []}

        for i in range(len(experimentalDS)):
            x = experimentalDS[i][0].to(config.DEVICE, dtype=torch.float)
            diameter = experimentalDS[i][1]

		# send the input to the device
            x = x.to(config.DEVICE)
            
            # img_h, img_w = 256, 256
            # center = (img_h-1)/2, (img_w-1)/2
            # Y, X = np.ogrid[:img_h, :img_w]
            # dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

            # mask = torch.from_numpy(dist_from_center) <= diameter/2
            # mask = mask*1  
            # mask = mask.to(config.DEVICE)
			# make the predictions and calculate the validation loss
            
            pred = unet(x.unsqueeze(0))
            # print(np.count_nonzero(pred.cpu().detach().numpy()))
            pred = torch.sigmoid(pred)
            # print(pred.cpu().detach().numpy())
            # pred = np.round(pred.squeeze().cpu().detach().numpy())
            pred = (pred.cpu().detach().numpy() > 0.1) #* 255

            # print(np.count_nonzero(pred))
            # pred_area = np.count_nonzero(pred)/(0.8*0.8*4**2)
            # pred_area = (np.count_nonzero(pred)*(0.8*0.8))/(4**2)
            
            pred_area = np.count_nonzero(pred)*(0.8**2)
            pred_area = pred_area/(4**2)
            
            # print(pred_area)
            pred_diameter = 2*np.sqrt(pred_area/math.pi)
            
            test_results["pred"].append(pred_diameter)
            test_results["gt"].append(diameter)  
            
            # totalTestLossExp += lossFunc(pred, mask.unsqueeze(0).unsqueeze(0).float())
        totalTestLossExp = mean_absolute_error(test_results["pred"], test_results["gt"])
        
	# calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgTestLoss = totalTestLoss / testSteps
    avgTestLossExp = totalTestLossExp# / exp_testSteps
        
    	# update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
    H["test_loss_exp"].append(avgTestLossExp)
        
    	# print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
    print("Train loss: {:.6f}, Test loss: {:.4f}, Experimental Test loss: {:.4f}".format(
        avgTrainLoss, avgTestLoss, avgTestLossExp))
    torch.save(unet, config.model_path(e+1))

# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.plot(H["test_loss_exp"], label="test_loss_exp")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
# plt.savefig(config.PLOT_PATH)
plt.show()

# serialize the model to disk
# torch.save(unet, config.MODEL_PATH)


# fig, ax1 = plt.subplots()
# plt.style.use("ggplot")
# ax1.set_xlabel('Epoch #')
# ax1.set_ylabel('Loss')
# ax1.plot(H["train_loss"], label="train_loss")
# ax1.plot(H["test_loss"], label="test_loss")

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('Test radius MAE (mm)', color='g')  # we already handled the x-label with ax1
# ax2.plot(H["test_loss_exp"], label="test_loss_exp", color = 'g')
# ax2.tick_params(axis='y', labelcolor='g')

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.title('Training losses and test MAE')
# plt.show()


fig, ax1 = plt.subplots(figsize=(10,5))
plt.style.use("ggplot")
ax1.set_xlabel('Epoch #')
ax1.set_ylabel('Loss')
ax1.plot(H["train_loss"], label="train_loss")
ax1.plot(H["test_loss"], label="test_loss")

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Test diameter MAE (mm)', color='g')  # we already handled the x-label with ax1
ax2.plot(H["test_loss_exp"], label="test_loss_exp", color = 'g')
ax2.tick_params(axis='y', labelcolor='g')
ax2.set_ylim(0,5)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Training losses and test MAE')
plt.savefig(config.PLOT_PATH, bbox_inches = 'tight')
plt.show()

print('Best performing network was iteration {} with {}'.format(np.argmin(H["test_loss_exp"]), H["test_loss_exp"][np.argmin(H["test_loss_exp"])]))