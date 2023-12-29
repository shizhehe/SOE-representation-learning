import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib as mpl
import sklearn.manifold
import sklearn.decomposition
import sklearn.cluster
import sklearn.svm
import h5py
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
from matplotlib import cm
import pandas as pd
import medutils
import torch

ckpt_path = 'ADNI/ckpt/ADNI/'
phase = 'test'
week = 18
task = 'pretext'
model = 'VN_Net'
model_label = 'VN_Net_fold_0_fixed_massive_normalized'
trained_data = h5py.File(ckpt_path+f'{model}/week{week}/{task}/{model_label}/result_{phase}/results_all.h5', 'r')

# setup data
img1 = trained_data['img1']
img2 = trained_data['img2']
label = trained_data['label']
rot = trained_data['rot']
axis = trained_data['axis']
z1 = trained_data['z1']
z2 = trained_data['z2']
z1_rot = trained_data['z1_matmul']
z2_rot = trained_data['z2_matmul']

sample_size = 64

random_sample = np.sort(np.random.choice(len(img1), sample_size, replace=False))
# sort
print(random_sample)
selected_img1 = img1[random_sample]
selected_img2 = img2[random_sample]
selected_labels = label[random_sample]
selected_rot = rot[random_sample]
selected_axis = axis[random_sample]
selected_z1 = z1[random_sample]
selected_z2 = z2[random_sample]
selected_z1_rot = z1_rot[random_sample]
selected_z2_rot = z2_rot[random_sample]


# filter
labels = trained_data['label'][:]
rot = trained_data['rot'][:]
axis = trained_data['axis'][:]

label_class = 0
angle = 90
filter_axis = [0, 1, 0]

# Define filter conditions
label_condition = (labels == label_class)
angle_condition = (rot == angle)
axis_condition = np.all(axis == filter_axis, axis=1)  # Assuming 'axis' is a 2D array

mask = label_condition & angle_condition & axis_condition

# Combine the conditions using logical AND
nc_img1 = img1[mask]
nc_img2 = img2[mask]
nc_rot = rot[mask]
nc_axis = axis[mask]
nc_z1 = z1[mask]
nc_z2 = z2[mask]
nc_z1_rot = z1_rot[mask]
nc_z2_rot = z2_rot[mask]

sample_size = min(len(nc_img1), 64)
print(f"Number of samples: {len(nc_img1)}")
random_sample = np.sort(np.random.choice(len(nc_img1), sample_size, replace=False))
# sort
print(random_sample)
nc_selected_img1 = nc_img1[random_sample]
nc_selected_img2 = nc_img2[random_sample]
nc_selected_rot = nc_rot[random_sample]
nc_selected_axis = nc_axis[random_sample]
nc_selected_z1 = nc_z1[random_sample]
nc_selected_z2 = nc_z2[random_sample]
nc_selected_z1_rot = nc_z1_rot[random_sample]
nc_selected_z2_rot = nc_z2_rot[random_sample]

# reshape each fs from (1024, 3) to (3*1024,)
nc_selected_z1 = nc_selected_z1.reshape(sample_size, -1)
nc_selected_z2 = nc_selected_z2.reshape(sample_size, -1)
nc_selected_z1_rot = nc_selected_z1_rot.reshape(sample_size, -1)
nc_selected_z2_rot = nc_selected_z2_rot.reshape(sample_size, -1)

print(np.std(nc_selected_z1[:, 0]))

# to visualize the feature space, we need to reduce the dimensionality using PCA and/or t-SNE
# first, let's try t-SNE
tsne = sklearn.manifold.TSNE(n_components=2, perplexity=10, random_state=0)
# what if all data is not fit in at the same time, all four feature spaces are fit in at the same time!!!

nc_z1_embedded = tsne.fit_transform(nc_selected_z1)
nc_z2_embedded = tsne.fit_transform(nc_selected_z2)
nc_z1_rot_embedded = tsne.fit_transform(nc_selected_z1_rot)
nc_z2_rot_embedded = tsne.fit_transform(nc_selected_z2_rot)

plt.figure(figsize=(10, 8))

plt.scatter(nc_z1_embedded[:, 0], nc_z1_embedded[:, 1], label='z1', marker='o', color='blue', s=5)
plt.scatter(nc_z2_embedded[:, 0], nc_z2_embedded[:, 1], label='z2', marker='o', color='red', s=5)
plt.scatter(nc_z1_rot_embedded[:, 0], nc_z1_rot_embedded[:, 1], label='z1_rot', marker='o', color='green', s=5)
plt.scatter(nc_z2_rot_embedded[:, 0], nc_z2_rot_embedded[:, 1], label='z2_rot', marker='o', color='purple', s=5)

plt.legend()
plt.title("Feature Distances Visualization")
plt.xlabel("Dimension 0")
plt.ylabel("Dimension 1")

plt.show()