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
from model import CLS
from so3_transformations.transformations import *
from util import *
import h5py
import matplotlib.pyplot as plt
import argparse
import random

LOCAL = True
DEBUG = False

phase = 'val'
week = 10

# set seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True

# cuda
cuda = 0
device = torch.device('cuda:'+ str(cuda))

if LOCAL:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

froze_encoder = False

# model setting
latent_size = 1024
use_feature = ['z']

pos_weight = [1]

# training setting
batch_size = 64
num_fold = 5
fold = 0
shuffle = True
aug = False # this could be leading to the incorrect shapes?! see implementation of CrossSectionalDataset


#model_id = "VN_Net_fold_2"
#model_id = "VN_Net_fold_3"
model_id = "baseline_rig_eval"
#model_name = 'VN_Net'
model_name = 'CLS'


data_type = 'single'
dataset_name = 'ADNI'
data_path = '/scratch/users/shizhehe/ADNI/'
if aug:
    img_file_name = 'ADNI_longitudinal_img_aug.h5'
else:
    img_file_name = 'ADNI_longitudinal_img.h5'
noimg_file_name = 'ADNI_longitudinal_noimg.h5'
subj_list_postfix = 'NC_AD'
if LOCAL:
    data_path = 'ADNI/'

# time
localtime = time.localtime(time.time())
time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)

# checkpoints
ckpt_folder = '/scratch/users/shizhehe/ADNI/ckpt/'
if LOCAL:
    ckpt_folder = 'ADNI/ckpt/'

ckpt_folder = '/scratch/users/shizhehe/ADNI/ckpt/'
if LOCAL:
    ckpt_folder = 'ADNI/ckpt/'

ckpt_path = os.path.join(ckpt_folder, dataset_name, model_name, f"week{week}", model_id)

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

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

print("Data loaded!!!")

# define model
if LOCAL:
    if model_name == 'VN_Net':
        model = VN_Net(latent_size, use_feature=use_feature, vn_module=True, encoder='base', dropout=(froze_encoder == False), gpu=None).to(device)
    elif model_name == 'CLS':
        model = CLS(latent_size, use_feature=use_feature, dropout=(froze_encoder == False), gpu=device).to(device)
else:
    if model_name == 'VN_Net':
        model = VN_Net(latent_size, use_feature=use_feature, vn_module=True, encoder='base', dropout=False, gpu=device).to(device)
    elif model_name == 'CLS':
        model = CLS(latent_size, use_feature=use_feature, dropout=(froze_encoder == False), gpu=device).to(device)

print("Model set!!!")

if froze_encoder:
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("Froze encoder!!!")

[model], start_epoch = load_checkpoint_by_key([model], ckpt_path, ['model'], device)

model.eval()

def evaluate_rotation(phase='val', set='val', save_res=True, info='', model=model, cpkt_path=ckpt_path):
    model.eval()

    if phase == 'val':
        loader = valDataLoader
    else:
        if set == 'train':
            loader = trainDataLoader
        elif set == 'val':
            loader = valDataLoader
        elif set == 'test':
            loader = testDataLoader
        else:
            raise ValueError('Undefined loader')

    zs_file_path = os.path.join(cpkt_path, 'result_train', 'results_allbatch.h5')
    if info == 'dataset':
        if not os.path.exists(zs_file_path):
            raise ValueError('Not existing zs for training batches!')
        else:
            zs_file = h5py.File(zs_file_path, 'r')
            zs_all = [torch.tensor(zs_file['z1']).to(device, dtype=torch.float), torch.tensor(zs_file['z2']).to(device, dtype=torch.float)]
            interval_all = torch.tensor(zs_file['interval']).to(device, dtype=torch.float)
    elif info == 'batch' and set == 'train' and os.path.exists(zs_file_path):
        return

    res_path = os.path.join(cpkt_path, 'result_rotation_'+set)
    print(res_path)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    path = os.path.join(res_path, 'results_rotation_all'+info+'.h5')
    if os.path.exists(path):
        raise ValueError('Exist results')
        os.remove(path)

    loss_all_dict = {'all': 0, 'recon': 0., 'dir': 0., 'dis': 0., 'cls': 0.}
    img2_list = []
    label_list = []
    age_list = []
    pred_list = []

    # possible rotations
    rotations = [90, 180, 270]
    axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    with torch.no_grad():
        for iter, sample in tqdm.tqdm(enumerate(loader, 0)):
            # rotate samples
            rotation = np.random.choice(rotations)
            rotation = random.randint(0, 360)
            axis = axes[np.random.randint(0, len(axes) - 1)]
            rot_mat = generate_rotation_matrix(axis, rotation, device) # R

            img1 = sample['img']#.to(device, dtype=torch.float) # img1 is the image at time t
            
            # img2 = rotated img1 by R
            img2 = rotate_batch(img1, batch_size, rot_mat)

            img1 = img1.to(device, dtype=torch.float).unsqueeze(1)
            img2 = img2.to(device, dtype=torch.float).unsqueeze(1)

            label = sample['label'].to(device, dtype=torch.float)

            # run model
            pred, _ = model.forward_single(img2)

            loss_cls, pred_sig = model.compute_classification_loss(pred, label, torch.tensor(pos_weight), dataset_name, subj_list_postfix)
            loss = loss_cls
            loss_all_dict['cls'] += loss_cls.item()

            pred_list.append(pred_sig.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

            if phase == 'test' and save_res:
                img2_list.append(img2.detach().cpu().numpy())
                age_list.append(sample['age'].numpy())

        for key in loss_all_dict.keys():
            loss_all_dict[key] /= (iter + 1)

        pred_list = np.concatenate(pred_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
        bacc = compute_classification_metrics(label_list, pred_list, dataset_name, subj_list_postfix)
        loss_all_dict['bacc'] = bacc

        if phase == 'test' and save_res:
            img2_list = np.concatenate(img2_list, axis=0)
            age_list = np.concatenate(age_list, axis=0)
            h5_file = h5py.File(path, 'w')
            h5_file.create_dataset('img1', data=img2_list)
            h5_file.create_dataset('label', data=label_list)
            h5_file.create_dataset('age', data=age_list)
            h5_file.create_dataset('pred', data=pred_list)

    return loss_all_dict

def rotate_batch(img1, batch_size, rot_mat):
    img1 = img1.requires_grad_(True)
    rotated_volumes = []
    for i in range(min(img1.shape[0], batch_size)):
        volume = img1[i]
        rotated_volumes.append(rotate_volume(volume, rot_mat, device, mode='trilinear'))
    img2 = torch.stack(rotated_volumes).to(device, dtype=torch.float)

    return img2

if phase == 'val':
    stat = evaluate_rotation(phase='test', set='test', save_res=True, model=model, cpkt_path=ckpt_path)