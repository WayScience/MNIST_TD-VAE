__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/12/17 16:45:38"

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import *
from prep_data import *
#import sys

with open("MNIST.pkl", 'rb') as file_handle:
    MNIST = pickle.load(file_handle)

data = MNIST_Dataset(MNIST['train_image'])


batch_size = 512
data_loader = DataLoader(data, batch_size = batch_size,
                         shuffle = True)


input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8
#create instance of TD_VAE class 
tdvae = TD_VAE(x_size = input_size, processed_x_size = processed_x_size, 
               b_size = belief_state_size, z_size = state_size)
# "CUDA is a parallel computing platform and application programming 
# interface (API) that allows software to use certain types of graphics
# processing units (GPUs) 

# "CUDA is a software layer that gives direct access to the GPU's 
# virtual instruction set and parallel computational elements, 
# for the execution of compute kernels"

# both from wikipedia 

# what is this doing? / why do we need to use it? 
tdvae = tdvae.cuda()

# Adam = an algorithm for first-order gradient-based optimization of
# stochastic (having a random probability distribution or pattern 
# that may be analyzed statistically but may not be predicted precisely) 
# objective functions, based on adaptive estimates of lower-order moments

# The method computes individual adaptive learning rates for
# different parameters from estimates of first and second 
# moments of the gradients
# paper: https://arxiv.org/pdf/1412.6980.pdf

# learning rate 0.0005 from original paper 
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)

# where does this file come from?
checkpoint = torch.load("./output/model/model_epoch_2999.pt")
tdvae.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# where do these numbers come from?
start_epoch = 3000
end_epoch = 6000
log_file_handle = open("./log/loginfo.txt", 'a')

for epoch in range(start_epoch, end_epoch):
    for idx, images in enumerate(data_loader):
        images = images.cuda()  
        tdvae.forward(images)
        t_1 = np.random.choice(16)
        t_2 = t_1 + np.random.choice([1,2,3,4])
        loss = tdvae.calculate_loss(t_1, t_2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()),
              file = log_file_handle, flush = True)
        
        #print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()))

    if (epoch + 1) % 50 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': tdvae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, "./output/model/model_epoch_{}.pt".format(epoch))
