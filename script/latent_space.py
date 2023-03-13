__author__ = "Keenan Manpearl"
__date__ = "2023/03/12"

"""
extract z-values from a trained model 
"""

import pickle
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.abspath("./"))
from prep_data import MNIST_Dataset
from model import TD_VAE, DBlock, PreProcess, Decoder

#### load trained model
checkpoint = torch.load("output_epochs/epoch_0.pt")
input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8
d_block_hidden_size = 50
decoder_hidden_size = 200
time_points = 20
batch_size = 512

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

data_file = "data/MNIST.pkl"
with open(data_file, "rb") as file_handle:
    MNIST = pickle.load(file_handle)

tdvae.eval()
tdvae = tdvae.cuda()

data = MNIST_Dataset(MNIST["train_image"], binary=False)

data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
idx, images = next(enumerate(data_loader))
images = images.cuda()
tdvae.forward(images)
z_values = tdvae.extract_latent_space(images, time_points)


z_df = pd.DataFrame(columns=["x", "y", "z", "vid"])
for time in range(time_points):
    for movie in range(len(z_values[2])):
        for z_val_idx in range(2 * state_size):
            x = time
            z = z_values[time][movie][z_val_idx]
            y = movie
            vid = z_val_idx
            row = pd.DataFrame({"x": [x], "y": [y], "z": [z], "vid": [vid]})
            z_df = pd.concat([z_df, row])


fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(projection="3d")

colors = {0: "orange", 1: "g"}
for i in range(2):
    ax.scatter(
        xs=z_df.loc[z_df.vid == i, "x"],
        ys=z_df.loc[z_df.vid == i, "y"],
        zs=z_df.loc[z_df.vid == i, "z"],
        color=colors[i],
    )

plt.savefig("latent_space_outputs/latent_space.png")
plt.show()