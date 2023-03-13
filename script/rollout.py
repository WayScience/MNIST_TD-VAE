__author__ = "Keenan Manpearl"
__date__ = "2023/1/24"

"""
original code by Xinqiang Ding <xqding@umich.edu>
After training the model, we can try to use the model to do jumpy predictions.
"""

import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from model import *
from prep_data import *


#### load trained model
checkpoint = torch.load("output_epochs/epoch_0.pt")
input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8
d_block_hidden_size = 50
decoder_hidden_size = 200

tdvae = TD_VAE(
    x_size=input_size,
    processed_x_size=processed_x_size,
    b_size=belief_state_size,
    z_size=state_size,
    d_block_hidden_size=d_block_hidden_size,
    decoder_hidden_size=decoder_hidden_size,
)

optimizer = optim.Adam(tdvae.parameters(), lr=0.0005)

tdvae.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

#### load dataset
with open("data/MNIST.pkl", "rb") as file_handle:
    MNIST = pickle.load(file_handle)
tdvae.eval()
tdvae = tdvae.cuda()

data = MNIST_Dataset(MNIST["train_image"], binary=False)
batch_size = 6
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
idx, images = next(enumerate(data_loader))

images = images.cuda()

## calculate belief
tdvae.forward(images)

## jumpy rollout
t1, t2 = 11, 15
rollout_images = tdvae.rollout(images, t1, t2)

#### plot results
fig = plt.figure(0, figsize=(12, 4))

fig.clf()
gs = gridspec.GridSpec(batch_size, t2 + 2)
gs.update(wspace=0.05, hspace=0.05)
for i in range(batch_size):
    for j in range(t1):
        axes = plt.subplot(gs[i, j])
        axes.imshow(1 - images.cpu().data.numpy()[i, j].reshape(28, 28), cmap="binary")
        axes.axis("off")

    for j in range(t1, t2 + 1):
        axes = plt.subplot(gs[i, j + 1])
        axes.imshow(
            1 - rollout_images.cpu().data.numpy()[i, j - t1].reshape(28, 28),
            cmap="binary",
        )
        axes.axis("off")

# fig.savefig("./output/rollout_result.eps")
plt.show()
sys.exit()
