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
import wandb

LOCAL = False
DEBUG = False
VNN = True

# set before every training!
feature_size = 48

week = 28
num_conv = 1
normalize_rotation = False
normalize_matrix = False
# week 16: don't freeze encoder
froze_encoder = False
#lr = 0.00002
#lr = 0.0001
lr = 0.001
#lr = 0.00001

ViT_dims = (768, 3, 8, 16)
phase = 'train'
task = "age"
encoder = 'ViT'

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

cuda = 0
device = torch.device('cuda:' + str(cuda))

if LOCAL:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-fold", "--fold", default=0)
parser.add_argument("-name", "--name", default="VN_Net_fold_{fold}_fixed_moderate_giga")

# Read arguments from command line
args = parser.parse_args()

fold = args.fold
base_name = args.name
print(f"{phase}-ing model on downstream task for fold {fold}")

model_name = 'VN_Net'
pretext_fold = 0
model_ckpt = f"{base_name.format(fold=fold)}_{'normalized' if normalize_matrix else 'unnormalized'}{'_frozen' if froze_encoder else ''}"
model_ckpt_pretext = f"{base_name.format(fold=pretext_fold)}_{'normalized' if normalize_matrix else 'unnormalized'}"

if encoder == 'base':
    latent_size = 1024
elif encoder == 'SWIN':
    latent_size = 384 * (feature_size // 24)
else:
    latent_size = 768

use_feature = ['z']

pos_weight = [1]

epochs = 50
batch_size = 64
if encoder == 'SWIN':
    batch_size = 4
elif encoder == 'ViT':
    batch_size = 32
num_fold = 5

shuffle = True
#lr = 0.01
#lr = 0.001
#lr = 0.0001
#aug = True
aug = False

data_type = 'single'
dataset_name = 'ADNI'
data_path = '/scratch/users/shizhehe/ADNI/'
if aug:
    img_file_name = 'ADNI_longitudinal_img_aug.h5'
else:
    img_file_name = 'ADNI_longitudinal_img.h5'
noimg_file_name = 'ADNI_longitudinal_noimg.h5'
subj_list_postfix = 'NC'
if LOCAL:
    data_path = 'ADNI/'

localtime = time.localtime(time.time())
time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)

ckpt_folder = '/scratch/users/shizhehe/ADNI/ckpt/'
if LOCAL:
    ckpt_folder = 'ADNI/ckpt/'

saved_path = os.path.join(ckpt_folder, dataset_name, model_name, f"week{week}", "pretext", model_ckpt_pretext)
ckpt_path = os.path.join(ckpt_folder, dataset_name, model_name, f"week{week}", "age", f"{model_ckpt}")
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
    if model_name == 'VN_Net':
        model = VN_Net(latent_size, use_feature=use_feature, vn_module=True, encoder=encoder, dropout=(froze_encoder == False), gpu=None, feature_size=feature_size, ViT_dims=ViT_dims).to(device)
else:
    if model_name == 'VN_Net':
        model = VN_Net(latent_size, use_feature=use_feature, vn_module=True, encoder=encoder, dropout=False, gpu=device, feature_size=feature_size, ViT_dims=ViT_dims).to(device)

print("Model set!!!")

if froze_encoder:
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("Froze encoder!!!")

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

if phase == 'train':
    [model], start_epoch = load_checkpoint_by_key([model], saved_path, ['model'], device)
else:
    [model], start_epoch = load_checkpoint_by_key([model], ckpt_path, ['model'], device)

subj_list_postfix = 'C_single'

if phase == 'train':
    print("Initialize Weights and Biases")
    wandb.init(
        # set the wandb project for run
        project="vector-neurons-mri-age",
        
        # track hyperparameters and run metadata
        config={
            "model_name": ckpt_path,
            "mode": phase,
            "train_mode": "downstream_age",
            "pretext_model": "best sofar",
            "week": week,
            "fold": fold,
            "architecture": model_name,
            "learning_rate": lr,
            "data_subset": subj_list_postfix,
            "epochs": epochs,
            "batch_size": batch_size,
            "notes": "downstream_age, new fixed rotation model with only NC and AD data samples"
        }
    )
    wandb.watch(model, log='all') # add gradient visualization

# ------- train -------
def train():
    print("Starting Training...")

    global_iter = 0
    monitor_metric_best = -1
    start_time = time.time()

    epoch_list = range(start_epoch + 1, start_epoch + epochs)
    # save losses for plotting
    train_dis_loss_list = []
    train_cls_loss_list = []
    train_total_loss_list = []

    train_r2_list = []
    val_r2_list = []

    val_dis_loss_list = []
    val_cls_loss_list = []
    val_total_loss_list = []
    
    print(f'Number of Training Batches: {len(trainDataLoader)}')

    # iterate through epochs
    for epoch in range(start_epoch + 1, start_epoch + epochs):
        model.train()
        loss_all_dict = {'all': 0, 'dis': 0., 'cls': 0.} # dis is distance loss, cls is classification loss, all is total loss
        global_iter0 = global_iter

        pred_list = [] 
        label_list = []

        # go through all data batches in dataset
        for iter, sample in enumerate(trainDataLoader, 0):
            global_iter += 1
            img1 = sample['img'].to(device, dtype=torch.float).unsqueeze(1)
            
            if subj_list_postfix == 'C_single':
                label = sample['age'].to(device, dtype=torch.float)
                mean_age = sample['mean_age'].to(device, dtype=torch.float)
                std_age = sample['std_age'].to(device, dtype=torch.float)
            else:
                label = sample['label'].to(device, dtype=torch.float)
                mean_age = None
                std_age = None

            if DEBUG:
                print(f"Input batch shape: {img1.shape}")
                print(f'Size: {img1.shape[0]}, batch_size: {batch_size}')

            # run model to fetch predictions
            pred, _ = model.forward_single(img1)
            
            # compute distance loss between R x z1 and z2
            loss_cls, pred_sig = model.compute_classification_loss(pred, label, torch.tensor(pos_weight), dataset_name, subj_list_postfix, task)
        
            loss = loss_cls

            loss_all_dict['cls'] += loss_cls.item()
            loss_all_dict['all'] += loss.item()

            if DEBUG:
                print(f'Sample classification loss: {loss_cls}')
                print(f'Sample total loss: {loss_cls}')

            pred_list.append(pred_sig.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

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
                # pdb.set_trace()
                print('Epoch[%3d], iter[%3d]: loss=[%.4f], cls=[%.4f], dis=[%.4f]' \
                        % (epoch, iter, loss.item(), loss_cls.item(), 0))

        #scheduler.step()

        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter
        save_result_stat(loss_all_dict, {'ckpt_path': ckpt_path}, info='epoch[%2d]'%(epoch))
        print(loss_all_dict)
        train_dis_loss_list.append(loss_all_dict['dis'])
        train_cls_loss_list.append(loss_all_dict['cls'])
        train_total_loss_list.append(loss_all_dict['all'])

        pred_list = np.concatenate(pred_list, axis=0) # pred already sigmoid
        label_list = np.concatenate(label_list, axis=0)
        train_stat = compute_classification_metrics(label_list, pred_list, dataset_name, subj_list_postfix, task, std_age, mean_age)
        train_r2_list.append(train_stat["r2"])

        # validation
        stat = evaluate(phase='val', set='val', save_res=False)
        monitor_metric = stat['r2']
        scheduler.step(monitor_metric)
        save_result_stat(stat, {'ckpt_path': ckpt_path}, info='val')
        print(stat)
        val_r2_list.append(stat['r2'])
        val_cls_loss_list.append(stat['cls'])
        val_total_loss_list.append(stat['all'])

        # save wandb
        wandb.log({
            "train_age_loss": loss_all_dict['cls'][0],
            "validation_age_loss": stat['cls'][0],
            "train_r2": train_stat['r2'],
            "validation_f2": stat['r2'][0],
            "train_mae": train_stat["mae"],
            "validation_mae": stat["mae"][0],
            "train_mse": train_stat["mse"],
            "validation_mse": stat["mse"][0]
        })

        # save ckp
        is_best = False
        print('Monitor metric: %.4f, Best monitor metric: %.4f' % (monitor_metric, monitor_metric_best))
        if monitor_metric >= monitor_metric_best:
            is_best = True
            monitor_metric_best = monitor_metric
            print('Best model so far found!')

        state = {'epoch': epoch, 'monitor_metric': monitor_metric, 'stat': stat, \
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
                'model': model.state_dict()}
        print(optimizer.param_groups[0]['lr'])
        #print(scheduler.get_last_lr())
        save_checkpoint(state, is_best, ckpt_path)
    
    # plot losses
    plt.plot(epoch_list, train_cls_loss_list, label='Training Cls. Loss')
    plt.plot(epoch_list, val_cls_loss_list, label='Validation Cls. Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Downstream Classification Training/Validation Losses')
    print(os.path.join(ckpt_path, 'separate_losses_downstream.png'))
    plt.savefig(os.path.join(ckpt_path, 'separate_losses_downstream.png'))
    plt.clf()

    plt.plot(epoch_list, train_total_loss_list, label='Training Total Loss')
    plt.plot(epoch_list, val_total_loss_list, label='Validation Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Downstream Total Losses')
    plt.savefig(os.path.join(ckpt_path, 'total_losses_downstream.png'))


def evaluate(phase='val', set='val', save_res=True, info='', wandb_log=False):
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

    loss_all_dict = {'all': 0, 'dis': 0., 'cls': 0.}
    img1_list = []
    img2_list = []
    label_list = []
    z1_list = []
    z2_list = []
    pred_list = []

    with torch.no_grad():
        for iter, sample in tqdm.tqdm(enumerate(loader, 0)):
            img1 = sample['img'].to(device, dtype=torch.float).unsqueeze(1)
            if subj_list_postfix == 'C_single':
                label = sample['age'].to(device, dtype=torch.float)
                mean_age = sample['mean_age'].to(device, dtype=torch.float)
                std_age = sample['std_age'].to(device, dtype=torch.float)
            else:
                label = sample['label'].to(device, dtype=torch.float)
                mean_age = None
                std_age = None
            # run model
            pred, _ = model.forward_single(img1)

            loss_cls, pred_sig = model.compute_classification_loss(pred, label, torch.tensor(pos_weight), dataset_name, subj_list_postfix, task)
            loss = loss_cls

            loss_all_dict['cls'] += loss_cls.item()
            loss_all_dict['all'] += loss.item()

            pred_list.append(pred_sig.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

            if phase == 'test' and save_res:
                img1_list.append(img1.detach().cpu().numpy())

        for key in loss_all_dict.keys():
            loss_all_dict[key] /= (iter + 1)

        pred_list = np.concatenate(pred_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
        stats = compute_classification_metrics(label_list, pred_list, dataset_name, subj_list_postfix, task, std_age, mean_age)
        loss_all_dict['r2'] = stats['r2']
        loss_all_dict['mae'] = stats['mae']
        loss_all_dict['mse'] = stats['mse']

        if phase == 'test' and save_res:
            img1_list = np.concatenate(img1_list, axis=0)
            h5_file = h5py.File(path, 'w')
            h5_file.create_dataset('img1', data=img1_list)
            h5_file.create_dataset('label', data=label_list)
            h5_file.create_dataset('pred', data=pred_list)

    if wandb_log:
        wandb.log({
            f"{phase}_loss": loss_all_dict['cls'], 
            f"{phase}_r2": loss_all_dict['r2'],
            f"{phase}_mse": loss_all_dict['mse'], 
            f"{phase}mae": loss_all_dict['mae']}
        )

    return loss_all_dict

if phase == 'train':
    train()
    [model], start_epoch = load_checkpoint_by_key([model], ckpt_path, ['model'], device)
    stat = evaluate(phase='test', set='test', save_res=True, info='batch', wandb_log=True)
else:
    stat = evaluate(phase='test', set='test', save_res=True, wandb_log=False)
    print(stat)
