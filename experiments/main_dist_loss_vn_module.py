# this is the main file for training the VN_Net model with distance loss and vn_module
# for the first part of the project, this will be used to train the model on the ADNI dataset based on the pretext task (SO3 equivariance)

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
from collections import defaultdict
import random
import wandb

LOCAL = True
DEBUG = False
VNN = True

phase = 'val'
encoder = 'base'
week = 18
normalize_rotation = False
normalize_matrix = False
num_conv = 1
weight_decay = 1e-4

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

cuda = 0
device = torch.device('cuda:' + str(cuda))

if LOCAL:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

froze_encoder = False

model_name = 'VN_Net'
latent_size = 512
#use_feature = ['z', 'delta_z']
use_feature = ['z']

pos_weight = [1]

epochs = 50
batch_size = 64
num_fold = 0

fold = 0 # modified for cross_validation

shuffle = True
#lr = 0.001
lr = 0.01
aug = False
#aug = True

data_type = 'single'
dataset_name = 'ADNI'
data_path = '/scratch/users/shizhehe/ADNI/'
img_file_name = 'ADNI_longitudinal_img.h5'
#img_file_name = 'ADNI_longitudinal_img_aug.h5'
noimg_file_name = 'ADNI_longitudinal_noimg.h5'
#subj_list_postfix = 'NC_AD'
subj_list_postfix = 'NC_AD_pMCI_sMCI'
if LOCAL:
    data_path = 'ADNI/'

localtime = time.localtime(time.time())
time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)

ckpt_folder = '/scratch/users/shizhehe/ADNI/ckpt/'
if LOCAL:
    ckpt_folder = 'ADNI/ckpt/'
#name = time_label + f"_fixed_rotation_fold_{fold}"
name = f"VN_Net_fold_{fold}_fixed_massive_{'normalized' if normalize_matrix else 'unnormalized'}"
name = f"VN_Net_best_results_prev"
if VNN:
    ckpt_path = os.path.join(ckpt_folder, dataset_name, model_name, f"week{week}", "pretext", name)
else:
    ckpt_path = os.path.join(ckpt_folder, dataset_name, model_name, f"week{week}", "pretext", name)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

print("Loading Data")

data_img = h5py.File(os.path.join(data_path, img_file_name), 'r')
data_noimg = h5py.File(os.path.join(data_path, noimg_file_name), 'r')

Data = LongitudinalData(dataset_name, data_path, img_file_name=img_file_name,
                        noimg_file_name=noimg_file_name, subj_list_postfix=subj_list_postfix,
                        data_type=data_type, batch_size=batch_size, num_fold=num_fold,
                        aug=aug, fold=fold, shuffle=shuffle)
trainDataLoader = Data.trainLoader
valDataLoader = Data.valLoader
testDataLoader = Data.testLoader

class_counts = defaultdict(int)

# Iterate over the dataset and count occurrences of each class
for loader in [trainDataLoader, valDataLoader, testDataLoader]:
    for sample in loader:
        labels = sample['label']
        for label in labels:
            label = label.item()
            class_counts[label] += 1

print(dict(class_counts))

print("Data loaded!!!")

if LOCAL:
    model = VN_Net(latent_size, use_feature=use_feature, vn_module=True, num_conv=num_conv, encoder=encoder, dropout=(froze_encoder == False), gpu=None, normalize_output=normalize_matrix).to(device)
else:
    model = VN_Net(latent_size, use_feature=use_feature, vn_module=True, num_conv=num_conv, encoder=encoder, dropout=(froze_encoder == False), gpu=device, normalize_output=normalize_matrix).to(device)

print("Model set!!!")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters in model: {total_params}")

if froze_encoder:
    for param in model.encoder.parameters():
        param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

if phase != 'train':
    [model], start_epoch = load_checkpoint_by_key([model], ckpt_path, ['model'], device)

model : VN_Net

# possible rotations
rotations = [90, 180, 270]
#rotations = [(0, 90), (90, 180), (180, 360)] # instead of fixed rotations, we will use random rotations
#rotations = [(0, 360)] # train once with only one rotation
axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

if phase == 'train':
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
            "model_params": total_params,
            "architecture": model_name,
            "learning_rate": lr,
            "data_subset": subj_list_postfix,
            "epochs": epochs,
            "batch_size": batch_size,
            "rotations": rotations,
            "num_conv": num_conv,
            "normalize_rotation_output": normalize_rotation,
            "normalize_matrix": normalize_matrix,
            "notes": "longer training, huge model, correct loss normalization, training with all samples, no matrix normalization or image normalization at all"
        }
    )

# ------- train -------
def train():
    wandb.watch(model, log='all') # add gradient visualization

    start_epoch = -1

    print("Starting Training...")

    global_iter = 0
    monitor_metric_best = 1000000
    start_time = time.time()

    epoch_list = []
    # save losses for plotting
    train_dis_loss_list = []
    train_total_loss_list = []

    val_dis_loss_list = []
    val_total_loss_list = []
    # iterate through epochs
    for epoch in range(start_epoch + 1, epochs):
        epoch_list.append(epoch)

        model.train()
        loss_all_dict = {'all': 0, 'dis': 0.} # dis is distance loss, all is total loss
        global_iter0 = global_iter

        print(f'Number of Training Batches: {len(trainDataLoader)}')

        # go through all data batches in dataset
        for iter, sample in enumerate(trainDataLoader, 0):
            #for _ in range(1):
            for rotation in rotations:
                if type(rotation) != int:
                    rotation = random.randint(rotation[0], rotation[1])
                #rotation = random.choice(rotations)   
                global_iter += 1
           
                axis = axes[np.random.randint(0, len(axes) - 1)]
                rot_mat = generate_rotation_matrix(axis, rotation, device)#.to(device, dtype=torch.float) # R
                inv_rot_mat = rot_mat.T # transpose of a matrix, and for a rotation matrix, this will give you the inverse of the rotation matrix

                img1 = sample['img']#.to(device, dtype=torch.float) # img1 is the image at time t

                if img1.shape[0] <= batch_size // 2:
                    print("Size of data too small!")
                    break
                
                # img2 = rotated img1 by R
                img2 = rotate_batch(img1, batch_size, rot_mat)

                # unsqueeze to add a dimension of size 1 at the specified position
                # because expected to be (batch_size, num_channels, depth, height, width)
                img1 = img1.to(device, dtype=torch.float).unsqueeze(1)
                img2 = img2.to(device, dtype=torch.float).unsqueeze(1)

                # run model to fetch feature spaces z1 and z2 of img1/img2
                z1 = model.forward_single_fs(img1)
                z2 = model.forward_single_fs(img2)

                z1_post_matmul = matmul_batch(z1, batch_size, rot_mat) # Rz1
                z2_post_matmul = matmul_batch(z2, batch_size, inv_rot_mat) # (R^-1)z2

                z1 = z1.to(device, dtype=torch.float)
                z2 = z2.to(device, dtype=torch.float)
                
                # compute distance loss between R x z1 and z2
                distance_loss = model.compute_distance_loss(z1, z1_post_matmul, z2, z2_post_matmul, z1.shape[0])

                loss = distance_loss

                loss_all_dict['dis'] += distance_loss.item()
                loss_all_dict['all'] += loss.item()

                if DEBUG:
                    print(f'Sample distance loss: {distance_loss}')
                    print(f'Sample total loss: {distance_loss}')

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                for name, param in model.named_parameters():
                    try:
                        if not torch.isfinite(param.grad).all():
                            pdb.set_trace()
                    except:
                        continue

                optimizer.step()
                optimizer.zero_grad()

                if global_iter % 1 == 0:
                    print('Epoch[%3d], iter[%3d]: loss=[%.4f], dis=[%.4f]' \
                            % (epoch, iter, loss.item(), distance_loss.item()))
                    
            #for key in loss_all_dict.keys(): # remember to normalize by rotation/matrix variations
            #    loss_all_dict[key] /= variations
            
        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter
        save_result_stat(loss_all_dict, {'ckpt_path': ckpt_path}, basename=f'stat_fold_{fold}.csv', info='epoch[%2d]'%(epoch))
        print(loss_all_dict)
        
        train_dis_loss_list.append(loss_all_dict['dis'])
        train_total_loss_list.append(loss_all_dict['all'])

        # validation
        stat = evaluate(phase='val', set='val', save_res=False)
        monitor_metric = stat['dis'] # set to loss
        scheduler.step(monitor_metric)
        save_result_stat(stat, {'ckpt_path': ckpt_path}, basename=f'stat_fold_{fold}.csv', info='val')
        print(stat)
        val_dis_loss_list.append(stat['dis'])
        val_total_loss_list.append(stat['all'])

        # save ckp
        is_best = False
        if monitor_metric <= monitor_metric_best:
            is_best = True
            monitor_metric_best = monitor_metric
            print('Improved model found! Saving checkpoint...')
        state = {'epoch': epoch, 'monitor_metric': monitor_metric, 'stat': stat, \
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
                'model': model.state_dict()}
        print(optimizer.param_groups[0]['lr'])
        save_checkpoint(state, is_best, ckpt_path)

        if phase == 'train':
            wandb.log({
                "train_dist_loss": loss_all_dict['dis'][0],
                "validation_dist_loss": stat['dis'][0]
            })

    
        # plot losses
        plt.plot(epoch_list, train_dis_loss_list, label='Training Dis. Loss')
        plt.plot(epoch_list, val_dis_loss_list, label='Validation Dis. Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title('Classification and Distance Training/Validation Losses')
        plt.savefig(os.path.join(ckpt_path, f'separate_losses_fold_{fold}.png'))
        plt.clf()

        plt.plot(epoch_list, train_total_loss_list, label='Training Total Loss')
        plt.plot(epoch_list, val_total_loss_list, label='Validation Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title('Total Losses')
        plt.savefig(os.path.join(ckpt_path, f'total_losses_fold_{fold}.png'))

    # compute training time
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def evaluate(phase='val', set='val', save_res=True, info=''):
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

    zs_file_path = os.path.join(ckpt_path, 'result_train', 'results_allbatch.h5')
    if info == 'dataset':
        if not os.path.exists(zs_file_path):
            raise ValueError('Not existing zs for training batches!')
        else:
            zs_file = h5py.File(zs_file_path, 'r')
            zs_all = [torch.tensor(zs_file['z1']).to(device, dtype=torch.float), torch.tensor(zs_file['z2']).to(device, dtype=torch.float)]
            interval_all = torch.tensor(zs_file['interval']).to(device, dtype=torch.float)
    elif info == 'batch' and set == 'train' and os.path.exists(zs_file_path):
        return

    res_path = os.path.join(ckpt_path, 'result_'+set)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    #path = os.path.join(res_path, 'results_all_no_training'+info+'.h5')
    path = os.path.join(res_path, 'results_all'+info+'.h5')
    if os.path.exists(path):
        # raise ValueError('Exist results')
        os.remove(path)

    loss_all_dict = {'all': 0, 'dis': 0.}
    img1_list = [] # unrotated
    img2_list = [] # rotated -> img1 x R
    label_list = []
    rot_list = []
    axis_list = []

    z1_list = [] # img1 -> model
    z2_list = [] # img2 -> model
    z1_matmul_list = [] # z1 x R
    z2_matmul_list = [] # z2 x R^-1

    age_list = []

    with torch.no_grad():
        for iter, sample in tqdm.tqdm(enumerate(loader, 0)):
            variations = 0

            for rotation in rotations:
                if type(rotation) != int:
                    rotation = random.randint(rotation[0], rotation[1])
                for axis in axes:
                    variations += 1

                    img1 = sample['img']#.to(device, dtype=torch.float)
                    label = sample['label'].to(device, dtype=torch.float)
                    #print(img1.shape)

                    rot_mat = generate_rotation_matrix(axis, rotation, device) # R
                    inv_rot_mat = rot_mat.T # transpose of a matrix, and for a rotation matrix, this will give you the inverse of the rotation matrix
            
                    # rotate batch: img2 = rotated img1 by R
                    img2 = rotate_batch(img1, batch_size, rot_mat)

                    img1 = img1.to(device, dtype=torch.float).unsqueeze(1)
                    img2 = img2.to(device, dtype=torch.float).unsqueeze(1)

                    # run model to fetch feature spaces z1 and z2 of img1/img2
                    z1 = model.forward_single_fs(img1)
                    z2 = model.forward_single_fs(img2)

                    z1_post_matmul = matmul_batch(z1, batch_size, rot_mat) # Rz1
                    z2_post_matmul = matmul_batch(z2, batch_size, inv_rot_mat) # (R^-1)z2

                    z1 = z1.to(device, dtype=torch.float)
                    z2 = z2.to(device, dtype=torch.float)
                    
                    # compute distance loss between R x z1 and z2
                    distance_loss = model.compute_distance_loss(z1, z1_post_matmul, z2, z2_post_matmul, z1.shape[0])
                    #print(distance_loss)
                    #print(z1.shape)
                    loss = distance_loss

                    loss_all_dict['dis'] += distance_loss.item()
                    loss_all_dict['all'] += loss.item()

                    if phase == 'test' and save_res:
                        label_list.append(label.detach().cpu().numpy())
                        img1_list.append(img1.detach().cpu().numpy())
                        img2_list.append(img2.detach().cpu().numpy())
                        rot_list.append([rotation] * img1.shape[0])
                        axis_list.append([axis] * img1.shape[0])
                        z1_list.append(z1.detach().cpu().numpy())
                        z2_list.append(z2.detach().cpu().numpy())
                        z1_matmul_list.append(z1_post_matmul.detach().cpu().numpy())
                        z2_matmul_list.append(z2_post_matmul.detach().cpu().numpy())
                
            for key in loss_all_dict.keys(): # remember to normalize by rotation/matrix variations
                loss_all_dict[key] /= variations

            if phase == 'test' and save_res:
                break

        for key in loss_all_dict.keys():
            loss_all_dict[key] /= (iter + 1)

        if phase == 'test' and save_res:
            label_list = np.concatenate(label_list, axis=0)
            img1_list = np.concatenate(img1_list, axis=0)
            img2_list = np.concatenate(img2_list, axis=0)
            rot_list = np.concatenate(rot_list, axis=0)
            axis_list = np.concatenate(axis_list, axis=0)
            z1_list = np.concatenate(z1_list, axis=0)
            z2_list = np.concatenate(z2_list, axis=0)
            z1_matmul_list = np.concatenate(z1_matmul_list, axis=0)
            z2_matmul_list = np.concatenate(z2_matmul_list, axis=0)
            
            h5_file = h5py.File(path, 'w')
            h5_file.create_dataset('label', data=label_list)
            h5_file.create_dataset('img1', data=img1_list)
            h5_file.create_dataset('img2', data=img2_list)
            h5_file.create_dataset('rot', data=rot_list)
            h5_file.create_dataset('axis', data=axis_list)
            h5_file.create_dataset('z1', data=z1_list)
            h5_file.create_dataset('z2', data=z2_list)
            h5_file.create_dataset('z1_matmul', data=z1_matmul_list)
            h5_file.create_dataset('z2_matmul', data=z2_matmul_list)
    print(loss_all_dict)
    return loss_all_dict

import medutils

def rotate_batch(img1, batch_size, rot_mat):
    #print(img1[0, :, :, :])
    #print("Mean:", torch.mean(img1[0, :, :, :]))
    #print("Min:", torch.min(img1[0, :, :, :]))
    #print("Max:", torch.max(img1[0, :, :, :]))
    #medutils.visualization.show(img1[0, :, :, :].cpu().numpy())
    #plt.show()

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

    #print(img2[0, :, :, :])
    #print("Mean:", torch.mean(img2[0, :, :, :]))
    #print("Min:", torch.min(img2[0, :, :, :]))
    #print("Max:", torch.max(img2[0, :, :, :]))
    #sys.exit()
    return img2

def matmul_batch(img1, batch_size, rot_mat, mode='batch'):
    img1 = img1.requires_grad_(True)
    img1 = img1.to(device, dtype=torch.float)
    rot_mat = rot_mat.to(device, dtype=torch.float)
    
    if mode == 'single':
        rotated_volumes = []
        for i in range(min(img1.shape[0], batch_size)):
            volume = img1[i]
            rotated_volumes.append(apply_matmul(volume, rot_mat))
        img2 = torch.stack(rotated_volumes).to(device, dtype=torch.float)
    elif mode == 'batch':
        img2 = img1 @ rot_mat
        img2 = img2.to(device, dtype=torch.float)
    else:
        raise ValueError('Invalid mode: {}'.format(mode))
    return img2

if phase == 'train':
    train()
else:
    stat = evaluate(phase='test', set='test', save_res=True)

if phase == 'train':
    # [optional] finish the wandb run
    wandb.finish()
