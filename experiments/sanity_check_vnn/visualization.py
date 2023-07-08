import medutils

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import glob
import nibabel as nib
from datetime import datetime

from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
import torch

#TEST_DATASET = ModelNetDataLoader(root='/Users/shizhehe/dev/research/vector_neurons_mri/vnn/data/modelnet40_normal_resampled', split='test')
#testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=4)

data_path = '/Users/shizhehe/dev/research/vector_neurons_mri/vnn/data/other/'
img_paths = glob.glob(data_path + '*.nii.gz')
img_paths = sorted(img_paths)

print("Loading {} image(s)".format(len(img_paths)))

# load single datafile
for img_path in img_paths[0:1]:
    img_nib = nib.load(os.path.join(img_path))
    img = img_nib.get_fdata()
    print("Image shape: {}".format(img.shape))

# transpose image
img = np.transpose(img, (2, 0, 1))


# ----- normalize image -----
img = medutils.visualization.normalize(img) # to 255 range 
#img = img - np.min(img)


# ----- slice image -----
img = img[0:32, 0:64, 0:64]


# ----- visualize volume -----
#medutils.visualization.imshow(img[5])
medutils.visualization.show(img)
plt.show()


# ----- visualize point cloud -----
#x, y, z = np.nonzero(img)
threshold = np.min(img[img > 0]) * 10
print("Threshold: {}".format(threshold))

#x, y, z = np.nonzero(img > 0)
x, y, z = np.nonzero(img > threshold)

marker_size = img[x, y, z] / (5 * np.max(img))  # Scale the marker size for visualization

# Convert voxel coordinates to world coordinates
world_coords = np.stack((x, y, z), axis=1)
print(world_coords.shape)

# Visualize point cloud with varying marker size based on intensity
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(world_coords[:,0], world_coords[:,1], world_coords[:,2], s=marker_size)
plt.show()



# sanity check rotation
print("Sanity check rotation...")
print(torch.rand(world_coords.shape[0])*360)

trot = RotateAxisAngle(angle=torch.rand(world_coords.shape[0])*360, axis="Z", degrees=True)

#points = trot.transform_points(torch.from_numpy(world_coords).float())
#print(torch.from_numpy(world_coords).float())