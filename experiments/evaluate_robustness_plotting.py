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
from so3_transformations.transformations import *
from util import *
import h5py
import matplotlib.pyplot as plt
import argparse
import random
import wandb

LOCAL = False
DEBUG = False

phase = 'val'
week = 21
num_conv = 1
normalize_matrix = False
normalize_rotation = False
froze_encoder = False

# set seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic=True

parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-fold", "--fold", default=0)
 
# Read arguments from command line
args = parser.parse_args()
 
fold = args.fold
print(f"{phase}-ing model on downstream task for fold {fold}")

# cuda
cuda = 0
device = torch.device('cuda:'+ str(cuda))

if LOCAL:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model setting
latent_size = 1024
use_feature = ['z']

pos_weight = [1]

# training setting
batch_size = 64
num_fold = 5
shuffle = True
aug = False 


#model_ckpt = f"VN_Net_fold_{fold}_best_results_prev_{'normalized' if normalize_matrix else 'unnormalized'}"
model_ckpt = f'baseline_robustness_small_fold_{fold}'
#model_ckpt = f'VN_Net_fold_{fold}_fixed_with_max_lowlr_special_small_milder_robustness_unnormalized_robustness'
#model_ckpt = f'VN_Net_fold_{fold}_fixed_with_max_lowlr_special_small_explicit_robustness_highalpha_unnormalized'
#model_ckpt = f'VN_Net_fold_{fold}_fixed_with_max_lowlr_special_small_explicit_discrete_lowalpha_robustness_unnormalized'
#model_ckpt = f'VN_Net_fold_{fold}_fixed_with_max_lowlr_special_small_robustness_unnormalized_robustness'


#model_name = 'VN_Net'
model_name = 'CLS'
normalize_matrix = False


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

ckpt_path = os.path.join(ckpt_folder, dataset_name, model_name, f"week{week}", "classification", f"{model_ckpt}{'_frozen' if froze_encoder else ''}")
plotting_path = os.path.join(ckpt_folder, dataset_name, model_name, 'plotting_data')

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

if not os.path.exists(plotting_path):
    os.makedirs(plotting_path)

print("Loading Data")

data_img = h5py.File(os.path.join(data_path, img_file_name), 'r')
data_noimg = h5py.File(os.path.join(data_path, noimg_file_name), 'r')

print("Initialize Weights and Biases")
wandb.init(
    # set the wandb project for run
    project="vector-neurons-mri",
    
    # track hyperparameters and run metadata
    config={
        "model_name": ckpt_path,
        "mode": phase,
        "train_mode": "pretext",
        "week": week,
        "fold": fold,
        "architecture": model_name,
        "data_subset": subj_list_postfix,
        "batch_size": batch_size,
        "num_conv": num_conv,
        "normalize_rotation_output": normalize_rotation,
        "normalize_matrix": normalize_matrix
    }
)

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
        model = VN_Net(latent_size, use_feature=use_feature, vn_module=True, num_conv=num_conv, encoder='base', dropout=(froze_encoder == False), gpu=None, normalize_output=normalize_matrix).to(device)
        model : VN_Net
    elif model_name == 'CLS':
        model = CLS(latent_size, use_feature=use_feature, dropout=(froze_encoder == False), gpu=None, num_conv=num_conv).to(device)
        model : CLS
else:
    if model_name == 'VN_Net':
        model = VN_Net(latent_size, use_feature=use_feature, vn_module=True, num_conv=num_conv, encoder='base', dropout=False, gpu=device, normalize_output=normalize_matrix).to(device)
        model : VN_Net
    elif model_name == 'CLS':
        model = CLS(latent_size, use_feature=use_feature, dropout=(froze_encoder == False), gpu=device, num_conv=num_conv).to(device)
        model : CLS

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

    # Iterate over the dataset and count occurrences of each class
    from collections import defaultdict
    class_counts = defaultdict(int)
    for sample in loader:
        labels = sample['label']
        for label in labels:
            label = label.item()
            class_counts[label] += 1

    print(dict(class_counts))

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
    pred_list = []  
    rot_list = []
    axis_list = []

    # possible rotations
    #rotations = [(0, 90), (90, 180), (180, 270), (270, 360)]
    #rotations = [(15, 30), (30, 45)]
    #rotations = [(10, 30), (35, 55), (170, 190)]
    #rotations = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180)]
    #rotations = [(0, 0), (90, 90), (180, 180), (270, 270)]
    rotations = [(rot, rot) for rot in range(0, 361, 15)] 
    axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # abstract dict based on rotations
    pred_list_rot = {f"{key[0]}-{key[1]}": [] for key in rotations}
    label_list_rot = {f"{key[0]}-{key[1]}": [] for key in rotations}

    bacc_list_rot = {}
    f1_list_rot = {}

    with torch.no_grad():
        for iter, sample in tqdm.tqdm(enumerate(loader, 0)):
            # rotate samples
            for axis in axes:
                for rotation in rotations:
                    rand_rotation = random.randint(rotation[0], rotation[1])
                    rot_mat = generate_rotation_matrix(axis, rand_rotation, device) # R

                    img1 = sample['img']#.to(device, dtype=torch.float) # img1 is the image at time t
                    
                    # img2 = rotated img1 by R
                    img2 = rotate_batch(img1, batch_size, rot_mat)

                    img1 = img1.to(device, dtype=torch.float).unsqueeze(1)
                    img2 = img2.to(device, dtype=torch.float).unsqueeze(1)

                    label = sample['label'].to(device, dtype=torch.float)

                    # run model
                    pred, _ = model.forward_single(img2)

                    loss_cls, pred_sig = model.compute_classification_loss(pred, label, torch.tensor(pos_weight), dataset_name, subj_list_postfix)

                    pred_list.append(pred_sig.detach().cpu().numpy())
                    label_list.append(label.detach().cpu().numpy())

                    # store results to different rotation levels
                    # abstract dict based on rotations
                    label_list_rot[f"{rotation[0]}-{rotation[1]}"].append(label.detach().cpu().numpy())
                    pred_list_rot[f"{rotation[0]}-{rotation[1]}"].append(pred_sig.detach().cpu().numpy())

                    if phase == 'test' and save_res:
                        img2_list.append(img2.detach().cpu().numpy())
                        rot_list.append([rotation] * img1.shape[0])
                        axis_list.append([axis] * img1.shape[0])

        # display results for different degrees of rotation:
        print("--- Performance per Rotation ---")
        for key in pred_list_rot.keys():
            print("----- Rotation: ", key)
            pred_list_rot[key] = np.concatenate(pred_list_rot[key], axis=0)
            label_list_rot[key] = np.concatenate(label_list_rot[key], axis=0)
            bacc = compute_classification_metrics(label_list_rot[key], pred_list_rot[key], dataset_name, subj_list_postfix)
            bacc_list_rot[key] = bacc['bacc']
            f1_list_rot[key] = bacc['f1']

        # save dict as csv
        import csv
        print(bacc_list_rot)
        print(f1_list_rot)
        with open(os.path.join(plotting_path, 'bacc_list_rot.csv'), 'a') as f:
            for key in bacc_list_rot.keys():
                f.write("%s,%s,%s\n"%(key,f1_list_rot[key], fold))

        with open(os.path.join(plotting_path, 'f1_list_rot.csv'), 'a') as f:
            for key in f1_list_rot.keys():
                f.write("%s,%s,%s\n"%(key,f1_list_rot[key], fold))

        print("--- Overall performance ---")
        label_list = np.concatenate(label_list, axis=0)
        pred_list = np.concatenate(pred_list, axis=0)
        bacc = compute_classification_metrics(label_list, pred_list, dataset_name, subj_list_postfix)
        loss_all_dict['bacc'] = bacc

        if phase == 'test' and save_res:
            img2_list = np.concatenate(img2_list, axis=0)
            rot_list = np.concatenate(rot_list, axis=0)
            axis_list = np.concatenate(axis_list, axis=0)
            h5_file = h5py.File(path, 'w')
            h5_file.create_dataset('img2', data=img2_list)
            h5_file.create_dataset('label', data=label_list)
            h5_file.create_dataset('pred', data=pred_list)
            h5_file.create_dataset('rotation', data=rot_list)
            h5_file.create_dataset('axis', data=axis_list)

    return loss_all_dict

def rotate_batch(img1, batch_size, rot_mat):
    img1 = img1.requires_grad_(True)
    rotated_volumes = []
    for i in range(min(img1.shape[0], batch_size)):
        volume = img1[i]
        # add normalization of rotated volume to linear transform into same range
        if normalize_rotation: 
            rotated_volumes.append(normalize_volume(volume, rotate_volume(volume, rot_mat, device, mode='trilinear')))
        else:
            rotated_volumes.append(rotate_volume(volume, rot_mat, device, mode='trilinear'))
        
    img2 = torch.stack(rotated_volumes).to(device, dtype=torch.float)

    return img2

if phase == 'val':
    stat = evaluate_rotation(phase='test', set='test', save_res=False, model=model, cpkt_path=ckpt_path)