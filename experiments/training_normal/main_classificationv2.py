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
from so3_transformations.transformations import *

import h5py

LOCAL = False
DEBUG = False
VNN = True

phase = 'train'
#phase = 'evaluate'

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
model_name = 'CLS'  
latent_size = 512
use_feature = ['z', 'delta_z']

pos_weight = [1]

# training setting
epochs = 40
batch_size = 64
num_fold = 5
fold = 0
shuffle = True
lr = 0.0001
aug = False # this could be leading to the incorrect shapes?! see implementation of CrossSectionalDataset

# dataset
data_type = 'single'
dataset_name = 'ADNI'
data_path = '/scratch/users/shizhehe/ADNI/'
img_file_name = 'ADNI_longitudinal_img.h5'
noimg_file_name = 'ADNI_longitudinal_noimg.h5'
#subj_list_postfix = 'NC_AD_pMCI_sMCI'
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
if VNN:
    ckpt_path = os.path.join(ckpt_folder, dataset_name, model_name, time_label + "_VNN" + "_v2")
else:
    ckpt_path = os.path.join(ckpt_folder, dataset_name, model_name, time_label + "_v2")
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
    model = CLS(latent_size, use_feature=use_feature, dropout=(froze_encoder == False), gpu=None).to(device)
else:
    model = CLS(latent_size, use_feature=use_feature, dropout=(froze_encoder == False), gpu=device).to(device)

print("Model set!!!")

# froze encoder
if froze_encoder:
    for param in model.encoder.parameters():
        param.requires_grad = False

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
# optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5)

start_epoch = -1

# possible rotations
rotations = [0, 90, 180, 270]
axes = [(0, 1), (0, 2), (1, 2)]

# ------- train -------
def train():
    print("Starting Training...")

    global_iter = 0
    monitor_metric_best = -1
    start_time = time.time()
    
    # iterate through epochs
    for epoch in range(start_epoch + 1, epochs):
        model.train()
        loss_all_dict = {'all': 0, 'recon': 0., 'dis': 0., 'dir': 0., 'cls': 0.}
        global_iter0 = global_iter

        pred_list = [] 
        label_list = []

        print(f'Number of Training Batches: {len(trainDataLoader)}')

        # go through all data batches in dataset
        for iter, sample in enumerate(trainDataLoader, 0):
            global_iter += 1

            img1 = sample['img'].cpu()

            if subj_list_postfix == 'C_single':
                label = sample['age'].to(device, dtype=torch.float)
            else:
                label = sample['label'].to(device, dtype=torch.float)

            # if remainder of data isn't enough for batch
            if DEBUG:
                print(f"Input batch shape: {img1.shape}")
                print(f'Size: {img1.shape[0]}, batch_size: {batch_size}')
            if img1.shape[0] <= batch_size // 2:
                print("Size of data too small!")
                break

            rotation = np.random.choice(rotations)
            axis = axes[np.random.randint(0, len(axes) - 1)]
            #axis = axes[0]
            rot_mat = generate_rotation_matrix(rotation) # R
        
            # rotate batch
            img2 = np.empty_like(img1)

            # I2 = R x I1
            for i in range(min(img1.shape[0], batch_size)):
                volume = img1[i]
                img2[i] = rotate_custom(volume, rot_mat, axis)
            img2 = torch.from_numpy(img2).to(device, dtype=torch.float) # I2

            img2 = img2.unsqueeze(1)
            
            # run model
            pred, z1 = model.forward_single(img2)

            if DEBUG:
                print(f'Prediction sample: {pred}')
                print(f'Label sample: {label}')

                print(f'Size of input: {img2.shape}, size of label: {label.shape}, size of prediction {pred.shape}')

            loss_cls, pred_sig = model.compute_classification_loss(pred, label, torch.tensor(pos_weight), dataset_name, subj_list_postfix)

            if DEBUG:
                print(f'Sample loss: {loss_cls}')

            #loss = config['lambda_cls'] * loss_cls
            loss = loss_cls
            loss_all_dict['cls'] += loss_cls.item()

            pred_list.append(pred_sig.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

            #if DEBUG:
                #print(pred_list)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            for name, param in model.named_parameters():
                try:
                    if not torch.isfinite(param.grad).all():
                        pdb.set_trace()
                except:
                    continue

            optimizer.step()
            optimizer.zero_grad()

            if global_iter % 1 == 0:
                # pdb.set_trace()
                print('Epoch[%3d], iter[%3d]: loss=[%.4f], cls=[%.4f]' \
                        % (epoch, iter, loss.item(), loss_cls.item()))

        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter
        save_result_stat(loss_all_dict, {'ckpt_path': ckpt_path}, info='epoch[%2d]'%(epoch))
        print(loss_all_dict)

        pred_list = np.concatenate(pred_list, axis=0) # pred already sigmoid
        label_list = np.concatenate(label_list, axis=0)
        compute_classification_metrics(label_list, pred_list, dataset_name, subj_list_postfix)

        # validation
        # pdb.set_trace()
        stat = evaluate(phase='val', set='val', save_res=False)
        monitor_metric = stat['bacc']
        scheduler.step(monitor_metric)
        save_result_stat(stat, {'ckpt_path': ckpt_path}, info='val')
        print(stat)

        # save ckp
        is_best = False
        if monitor_metric >= monitor_metric_best:
            is_best = True
            monitor_metric_best = monitor_metric
        state = {'epoch': epoch, 'monitor_metric': monitor_metric, 'stat': stat, \
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
                'model': model.state_dict()}
        print(optimizer.param_groups[0]['lr'])
        save_checkpoint(state, is_best, ckpt_path)


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
    path = os.path.join(res_path, 'results_all'+info+'.h5')
    if os.path.exists(path):
        # raise ValueError('Exist results')
        os.remove(path)

    loss_all_dict = {'all': 0, 'recon': 0., 'dir': 0., 'dis': 0., 'cls': 0.}
    img1_list = []
    label_list = []
    recon1_list = []
    z1_list = []
    age_list = []
    pred_list = []

    with torch.no_grad():
        for iter, sample in tqdm.tqdm(enumerate(loader, 0)):
            img1 = sample['img'].to(device, dtype=torch.float).unsqueeze(1)
            if subj_list_postfix == 'C_single':
                label = sample['age'].to(device, dtype=torch.float)
            else:
                label = sample['label'].to(device, dtype=torch.float)

            # run model
            pred, z1 = model.forward_single(img1)

            loss_cls, pred_sig = model.compute_classification_loss(pred, label, torch.tensor(pos_weight), dataset_name, subj_list_postfix)
            #loss = config['lambda_cls'] * loss_cls
            loss = loss_cls
            loss_all_dict['cls'] += loss_cls.item()

            pred_list.append(pred_sig.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

            if phase == 'test' and save_res:
                img1_list.append(img1.detach().cpu().numpy())
                age_list.append(sample['age'].numpy())

        for key in loss_all_dict.keys():
            loss_all_dict[key] /= (iter + 1)

        pred_list = np.concatenate(pred_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
        bacc = compute_classification_metrics(label_list, pred_list, dataset_name, subj_list_postfix)
        loss_all_dict['bacc'] = bacc

        if phase == 'test' and save_res:
            img1_list = np.concatenate(img1_list, axis=0)
            age_list = np.concatenate(age_list, axis=0)
            h5_file = h5py.File(path, 'w')
            h5_file.create_dataset('img1', data=img1_list)
            h5_file.create_dataset('label', data=label_list)
            h5_file.create_dataset('age', data=age_list)
            h5_file.create_dataset('pred', data=pred_list)


    return loss_all_dict

if phase == 'train':
    train()
else:
    stat = evaluate(phase='test', set='test', save_res=True)
