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
from so3_transformations.transformations_non_grad import *
from util import *
import h5py
import matplotlib.pyplot as plt

LOCAL = False
DEBUG = False
VNN = True

phase = 'train'

lambda_cls = 0.005

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

model_name = 'CLS'
latent_size = 512
use_feature = ['z', 'delta_z']

pos_weight = [1]

epochs = 40
batch_size = 64
num_fold = 5
fold = 0
shuffle = True
lr = 0.0001
aug = False

data_type = 'single'
dataset_name = 'ADNI'
data_path = '/scratch/users/shizhehe/ADNI/'
img_file_name = 'ADNI_longitudinal_img.h5'
noimg_file_name = 'ADNI_longitudinal_noimg.h5'
subj_list_postfix = 'NC_AD'
if LOCAL:
    data_path = 'ADNI/'

localtime = time.localtime(time.time())
time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)

ckpt_folder = '/scratch/users/shizhehe/ADNI/ckpt/'
if LOCAL:
    ckpt_folder = 'ADNI/ckpt/'
if VNN:
    ckpt_path = os.path.join(ckpt_folder, dataset_name, model_name, time_label + "_DIST_VNN")
else:
    ckpt_path = os.path.join(ckpt_folder, dataset_name, model_name, time_label)
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

from collections import defaultdict
# Initialize a dictionary to store class counts
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
    model = CLS(latent_size, use_feature=use_feature, dropout=(froze_encoder == False), gpu=None).to(device)
else:
    #model = CLS(latent_size, use_feature=use_feature, dropout=(froze_encoder == False), gpu=device).to(device)
    model = CLS(latent_size, use_feature=use_feature, dropout=False, gpu=device).to(device)

print("Model set!!!")

if froze_encoder:
    for param in model.encoder.parameters():
        param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

start_epoch = -1

# possible rotations
rotations = [180] # should I remove 0? i think 0 is causing the issues, 0 removed for training
axes = [(0, 1), (0, 2), (1, 2)]

# ------- train -------
def train():
    print("Starting Training...")

    global_iter = 0
    monitor_metric_best = -1
    start_time = time.time()

    epoch_list = range(start_epoch + 1, epochs)
    # save losses for plotting
    train_dis_loss_list = []
    train_cls_loss_list = []
    train_total_loss_list = []

    val_dis_loss_list = []
    val_cls_loss_list = []
    val_total_loss_list = []
    
    # iterate through epochs
    for epoch in range(start_epoch + 1, epochs):
        model.train()
        loss_all_dict = {'all': 0, 'dis': 0., 'cls': 0.} # dis is distance loss, cls is classification loss, all is total loss
        global_iter0 = global_iter

        print(f'Number of Training Batches: {len(trainDataLoader)}')

        # go through all data batches in dataset
        for iter, sample in enumerate(trainDataLoader, 0):
            global_iter += 1
            img1 = sample['img'].detach().cpu().numpy()

            if DEBUG:
                print(f"Input batch shape: {img1.shape}")
                print(f'Size: {img1.shape[0]}, batch_size: {batch_size}')
            if img1.shape[0] <= batch_size // 2:
                print("Size of data too small!")
                break
           
            # img2 = rotated img1 by R
            rotation = np.random.choice(rotations)
            axis = axes[np.random.randint(0, len(axes) - 1)]
            #axis = axes[0]

            rot_mat = generate_rotation_matrix(rotation) # R
            inv_rot_mat = rot_mat.T # transpose of a matrix, and for a rotation matrix, this will give you the inverse of the rotation matrix
            # rotate batch
            img2 = rotate_batch(img1, batch_size, rot_mat, axis)

            img1 = torch.from_numpy(img1).to(device, dtype=torch.float).unsqueeze(1)
            img2 = img2.unsqueeze(1)

            # run model to fetch feature spaces z1 and z2 of img1/img2
            z1 = model.forward_single_fs(img1)
            print(z1.shape)
            z2 = model.forward_single_fs(img2)

            # not entirely sure if this preserves the gradient
            z1_post_rot = rotate_batch(z1.detach().cpu().numpy(), batch_size, rot_mat, axis, one_dim = True) # Rz1
            z2_post_rot = rotate_batch(z2.detach().cpu().numpy(), batch_size, inv_rot_mat, axis, one_dim = True) # (R^-1)z1

            z1 = z1.to(device, dtype=torch.float)
            z2 = z2.to(device, dtype=torch.float)
            
            # compute distance loss between R x z1 and z2
            distance_loss = model.compute_distance_loss(z1, z1_post_rot, z2, z2_post_rot, batch_size)

            loss = distance_loss

            loss_all_dict['dis'] += distance_loss.item()
            loss_all_dict['all'] += loss.item()

            if DEBUG:
                print(f'Sample distance loss: {distance_loss}')
                print(f'Sample total loss: {distance_loss}')

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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
            #sys.exit()

        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter
        save_result_stat(loss_all_dict, {'ckpt_path': ckpt_path}, info='epoch[%2d]'%(epoch))
        print(loss_all_dict)
        train_dis_loss_list.append(loss_all_dict['dis'])
        train_total_loss_list.append(loss_all_dict['all'])

        # validation
        stat = evaluate(phase='val', set='val', save_res=False)
        monitor_metric = stat['dis'] # set to loss
        scheduler.step(monitor_metric)
        save_result_stat(stat, {'ckpt_path': ckpt_path}, info='val')
        print(stat)
        val_dis_loss_list.append(stat['dis'])
        val_total_loss_list.append(stat['all'])

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
    
    # plot losses
    plt.plot(epoch_list, train_dis_loss_list, label='Training Dis. Loss')
    plt.plot(epoch_list, val_dis_loss_list, label='Validation Dis. Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Classification and Distance Training/Validation Losses')
    print(os.path.join(ckpt_path, 'separate_losses.png'))
    plt.savefig(os.path.join(ckpt_path, 'separate_losses.png'))
    plt.clf()

    plt.plot(epoch_list, train_total_loss_list, label='Training Total Loss')
    plt.plot(epoch_list, val_total_loss_list, label='Validation Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Total Losses')
    plt.savefig(os.path.join(ckpt_path, 'total_losses.png'))

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
    path = os.path.join(res_path, 'results_all'+info+'.h5')
    if os.path.exists(path):
        # raise ValueError('Exist results')
        os.remove(path)

    loss_all_dict = {'all': 0, 'dis': 0.}
    img1_list = []
    age_list = []

    with torch.no_grad():
        for iter, sample in tqdm.tqdm(enumerate(loader, 0)):
            img1 = sample['img'].detach().cpu().numpy()

            print(img1.shape)

            # img2 = rotated img1 by R
            rotation = np.random.choice(rotations)
            #axis = axes[np.random.randint(0, len(axes) - 1)]
            axis = axes[0]
            
            rot_mat = generate_rotation_matrix(rotation) # R
            inv_rot_mat = rot_mat.T # transpose of a matrix, and for a rotation matrix, this will give you the inverse of the rotation matrix
            # rotate batch
            img2 = rotate_batch(img1, batch_size, rot_mat, axis)

            img1 = torch.from_numpy(img1).to(device, dtype=torch.float).unsqueeze(1)
            img2 = img2.unsqueeze(1)

            # run model to fetch feature spaces z1 and z2 of img1/img2
            z1 = model.forward_single_fs(img1)
            print(z1.shape)
            z2 = model.forward_single_fs(img2)

            # not entirely sure if this preserves the gradient
            z1_post_rot = rotate_batch(z1.detach().cpu().numpy(), batch_size, rot_mat, axis, one_dim = True) # Rz1
            z2_post_rot = rotate_batch(z2.detach().cpu().numpy(), batch_size, inv_rot_mat, axis, one_dim = True) # (R^-1)z1

            z1 = z1.to(device, dtype=torch.float)
            z2 = z2.to(device, dtype=torch.float)
            
            # compute distance loss between R x z1 and z2
            distance_loss = model.compute_distance_loss(z1, z1_post_rot, z2, z2_post_rot, batch_size)

            loss = distance_loss

            loss_all_dict['dis'] += distance_loss.item()
            loss_all_dict['all'] += loss.item()

            if phase == 'test' and save_res:
                img1_list.append(img1.detach().cpu().numpy())
                age_list.append(sample['age'].numpy())

        for key in loss_all_dict.keys():
            loss_all_dict[key] /= (iter + 1)

        #bacc = compute_classification_metrics(label_list, pred_list, dataset_name, subj_list_postfix)

        if phase == 'test' and save_res:
            img1_list = np.concatenate(img1_list, axis=0)
            age_list = np.concatenate(age_list, axis=0)
            h5_file = h5py.File(path, 'w')
            h5_file.create_dataset('img1', data=img1_list)
            h5_file.create_dataset('age', data=age_list)

    return loss_all_dict

# have to preserve gradient
def rotate_batch(img1, batch_size, rot_mat, axis, one_dim = False):
    img2 = np.empty_like(img1)
    if one_dim:
        img2 = np.expand_dims(img2, axis=1)
    #img1 = torch.tensor(img1, requires_grad=True) # I1
    img1 = torch.from_numpy(img1)#.to(device, dtype=torch.float) # I1
    for i in range(min(img1.shape[0], batch_size)):
        volume = img1[i]
        if one_dim:
            volume = volume.unsqueeze(0)
        img2[i] = rotate_custom(volume, rot_mat, axis)
    img2 = torch.from_numpy(img2).to(device, dtype=torch.float) # I2
    #img2 = torch.tensor(img2, requires_grad=True).to(device, dtype=torch.float) # I2
    if one_dim:
        img2 = torch.squeeze(img2, axis=1)
    return img2

if phase == 'train':
    train()
else:
    stat = evaluate(phase='test', set='test', save_res=True)
