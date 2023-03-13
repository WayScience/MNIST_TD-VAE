__author__ = "Keenan Manpearl"
__date__ = "2023/03/12"

"""
extract z-values from a trained model 
"""

import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os 
sys.path.append(os.path.abspath('./'))
from prep_data import Mitocheck_Dataset
from model import TD_VAE, DBlock, PreProcess, Decoder

#### load trained model
checkpoint = torch.load("output/compression10_epoch_49.pt")
input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8
d_block_hidden_size = 50
decoder_hidden_size = 200
time_points = 20

tdvae = TD_VAE(
    x_size=input_size,
    processed_x_size=processed_x_size,
    b_size=belief_state_size,
    z_size=state_size,
    d_block_hidden_size=d_block_hidden_size,
    decoder_hidden_size=decoder_hidden_size,
)

optimizer = optim.Adam(tdvae.parameters(), lr=0.00005)

tdvae.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

data_file = "mitocheck_compression10_2movies.pkl"
with open(data_file, "rb") as file_handle:
    mitocheck = pickle.load(file_handle)

tdvae.eval()
tdvae = tdvae.cuda()

data = Mitocheck_Dataset(mitocheck)
batch_size = 512
data_loader = DataLoader(data, batch_size = batch_size, shuffle=True)
idx, images = next(enumerate(data_loader))
images = images.cuda()
tdvae.forward(images)
z_values = tdvae.extract_latent_space(images, time_points)