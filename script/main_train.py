__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/12/17 16:45:38"

import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader #, Dataset
from model import *
from prep_data import *

#### preparing dataset
with open("MNIST.pkl", 'rb') as file_handle:
    MNIST = pickle.load(file_handle)

data = MNIST_Dataset(MNIST['train_image'])


# 512 from original paper
# batch size means we are doing mini-batch gradient descent
# second most important for overall performance 
# mini batch gradient descent 
# default method for implementing gradient descent in deep learning
# computationally effecient, stable convergance, faster learning 
# LSTM = long short-term meomry
# recurrent neural network that can process entire sequences of data
batch_size = 512
data_loader = DataLoader(data,
                         batch_size = batch_size,
                         shuffle = True)

#### build a TD-VAE model
# from original paper: 
# tate size = 8 
# Belief states = 50
# 28*28 (dimensions of images) = 784 

input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8
# initate instance of class
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)

# CUDA is a software layer that gives direct access to the GPU's 
# virtual instruction set and parallel computational elements, 
# for the execution of compute kernels
tdvae = tdvae.cuda() 

#### training

# learning rate = 0.0005 from original paper 
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)
num_epoch = 6000
log_file_handle = open("loginfo.txt", 'w')
for epoch in range(num_epoch):
    for idx, images in enumerate(data_loader):        
        images = images.cuda()  
        tdvae.forward(images)
        # 20 total time points so why stop at 16
        t_1 = np.random.choice(16)
        # add a number between 1 and 4 to get t2
        t_2 = t_1 + np.random.choice([1,2,3,4])
        loss = tdvae.calculate_loss(t_1, t_2)
        # stores gradient from old batch
        # must set gradients to 0 for each new batch
        optimizer.zero_grad()
        # computes the derivative of the loss w.r.t. 
        # the parameters using backpropagation.
        loss.backward()
        # updates parameteres based on current gradient
        optimizer.step()

        print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()),
              file = log_file_handle, flush = True)
        
        print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()))

    if (epoch + 1) % 50 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': tdvae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, "output/model_epoch_{}.pt".format(epoch))


for i in range(len(params)):
    print(params[i].size())