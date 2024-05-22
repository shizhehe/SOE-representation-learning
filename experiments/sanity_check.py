import os
import glob
import sys
import time
import torch
import torch.optim as optim
import numpy as np
import yaml
import pdb
import tqdm
from model_vn import *
from util import *
import matplotlib.pyplot as plt
import wandb
import h5py
from so3_transformations.transformations import *
import medutils
from skimage.transform import resize

fold = 0
dataset_name = 'RADFUSION'

# set seed
seed = 10

# training setting
epochs = 50
batch_size = 16
num_fold = 5
shuffle = True
#aug = True
aug = False

"""# cuda
cuda = 0
device = torch.device('cuda:'+ str(cuda))

# dataset
data_type = 'single'
data_path = '/scratch/users/shizhehe/ADNI/'
if dataset_name == 'RADFUSION':
    data_path = '/scratch/groups/kpohl/temp_radfusion/multimodalpulmonaryembolismdataset/RADFUSION/'
if aug:
    img_file_name = f'{dataset_name}_longitudinal_img_aug.h5'
else:
    img_file_name = f'{dataset_name}_longitudinal_img.h5'
noimg_file_name = f'{dataset_name}_longitudinal_noimg.h5'
subj_list_postfix = 'NC_AD'
if dataset_name == 'RADFUSION':
    subj_list_postfix = 'NC_PE'

print("Loading Data")

data_img = h5py.File(os.path.join(data_path, img_file_name), 'r')
data_noimg = h5py.File(os.path.join(data_path, noimg_file_name), 'r')

# define dataset
Data = LongitudinalData(dataset_name, data_path, img_file_name=img_file_name,
            noimg_file_name=noimg_file_name, subj_list_postfix=subj_list_postfix,
            data_type=data_type, batch_size=batch_size, num_fold=num_fold,
            aug=aug, fold=fold, shuffle=shuffle)
trainDataLoader = Data.trainLoader
valDataLoader = Data.valLoader
testDataLoader = Data.testLoader

from collections import defaultdict
# Initialize a dictionary to store class counts
class_counts = defaultdict(int)

print(f'{len(trainDataLoader)}')
# Iterate over the dataset and count occurrences of each class
for loader in [trainDataLoader, valDataLoader, testDataLoader]:
    for sample in tqdm.tqdm(loader):
        labels = sample['label']
        img1 = sample['img'].to(device, dtype=torch.float)[0]
        label = labels[0].item()
        break"""


img1 = np.load('/Users/shizhehe/dev/research/vector_neurons_mri/RADFUSION/1/1419.npy')
img1 = resize(img1, (128, 128, 128), mode='constant', anti_aliasing=True)
img1 = torch.tensor(img1, dtype=torch.float32)
label = 0
device = torch.device('cpu')

print(f'img1 shape: {img1.shape}')
print(f'label: {label}')

"""# save slice 60
img1_np = img1.cpu().numpy()
img1_np = img1_np[60:61]
print(f'img1 shape: {img1_np.shape}')
plt.imsave('slice60.png', img1_np[0], cmap='gray')"""

print(f'img norm: {torch.norm(img1)}')

medutils.visualization.show(img1.detach().numpy())
plt.show()

# rotate and save
degree = 90

axis = [0, 0, 1]
rot_mat = generate_rotation_matrix(axis, degree, device)
print(f'rot_mat: {rot_mat}')
rotated = rotate_volume(img1, rot_mat, device, mode='trilinear')
print(f'rotated shape: {rotated.shape}')
medutils.visualization.show(rotated.detach().numpy())
plt.show()

#plt.imsave('slice60_rotated.png', rotated[0], cmap='gray')