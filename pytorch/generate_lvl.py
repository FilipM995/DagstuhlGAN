import json
from matplotlib import cm, pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from models import dcgan
from torch.autograd import Variable
from preprocess_level import print_from_mapping

examples_json = "pytorch/zelda.json"
original_images = np.array(json.load(open(examples_json)))
model_to_load = "pytorch/bin_no_walls/netG_epoch_14900_0_32.pth"
# model_to_load = "pytorch/experiment_1/netG_epoch_9950_0_32.pth"

nz = 32

batchSize = 10
# nz = 10 #Dimensionality of latent vector

imageSize = 16
ngf = 64
ngpu = 1
n_extra_layers = 0

one_tile_type = True

z_dims = 2 if one_tile_type else 10  # number different titles

generator = dcgan.DCGAN_G(imageSize, nz, z_dims, ngf, ngpu, n_extra_layers)
# generator.load_state_dict(torch.load('netG_epoch_24.pth', map_location=lambda storage, loc: storage))
generator.load_state_dict(torch.load(
    model_to_load, map_location=lambda storage, loc: storage))

lv = np.random.normal(0, 1, (batchSize, nz))

latent_vector = torch.FloatTensor(lv).view(batchSize, nz, 1, 1)

# normalize the latent vector to [-1,1]


def normalize(x):
    return x/(np.sqrt(1+np.power(x, 2)))


# latent_vector = normalize(latent_vector)
with torch.no_grad():
    levels = generator(Variable(latent_vector))

# levels.data = levels.data[:,:,:14,:28] #Cut of rest to fit the 14x28 tile dimensions

level = levels.data.cpu().numpy()
# Cut of rest to fit the 14x28 tile dimensions

original_shape = (12, 7)
pad_height = (level.shape[2] - original_shape[0]) // 2
pad_width = (level.shape[3] - original_shape[1]) // 2

level = level[:, :, pad_height:-pad_height, pad_width:-pad_width-1]

level = np.argmax(level, axis=1)

original_images = original_images[torch.randperm(len(original_images))]

original_images = original_images[:batchSize, :, :]

if one_tile_type:
    original_images[original_images == 2] = -1
    original_images[original_images != -1] = 0
    original_images[original_images == -1] = 1


# levels.data[levels.data > 0.] = 1  #SOLID BLOCK
# levels.data[levels.data < 0.] = 2  #EMPTY TILE

# Jacob: Only output first level, since we are only really evaluating one at a time

mapping = {'W': 0, '-': 1, 'F': 2, 'B': 3, 'M': 4,
           'P': 5, 'O': 6, 'I': 7, 'D': 8, 'S': 9}

print_from_mapping(level[0])

# Define discrete numbers from 0 to 9
colors = ["blue", "cyan", "green", "yellow", "orange",
          "red", "purple", "pink", "gray", "black"]
custom_cmap = ListedColormap(colors)

# Visualize discrete colormap
fig, ax = plt.subplots(figsize=(8, 2))
for i, color in enumerate(colors):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))

ax.set_xticks(np.arange(10) + 0.5)
ax.set_xticklabels(np.arange(10))
ax.set_yticks([])
ax.set_xlim(0, 10)
ax.set_title("Custom Discrete Colormap (0-9)")

# Plot 10 original images
fig, axs = plt.subplots(2, 5)
for i in range(2):
    for j in range(5):
        axs[i, j].imshow(original_images[i*5+j],
                         cmap=custom_cmap, interpolation='nearest')
        axs[i, j].axis('off')
plt.suptitle('Original Images')
plt.savefig('original_images.png')

# Plot 10 generated images
fig, axs = plt.subplots(2, 5)
for i in range(2):
    for j in range(5):
        axs[i, j].imshow(level[i*5+j], cmap=custom_cmap, interpolation='nearest')
        axs[i, j].axis('off')
plt.suptitle('Generated Images')
plt.savefig('generated_images.png')
# # Plot the array using a colormap
# plt.imshow(level[0], cmap='rainbow', interpolation='nearest')
# plt.colorbar()
# plt.title('Array Plot')
# plt.show()

# print color number mapping

# Create a figure and axis
