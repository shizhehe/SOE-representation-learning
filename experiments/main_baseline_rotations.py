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
from so3_transformations.transformations import *
import argparse
import wandb
import h5py
import random

LOCAL = False
DEBUG = False
VNN = False

parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-fold", "--fold", default=0)
# Read arguments from command line
args = parser.parse_args()
fold = args.fold
#fold = 0

phase = 'train'
# model setting
week = 22
num_conv = 1
model_ckpt = f'baseline_robustness_one_rotation_milder_small_fold_{fold}'
lr = 0.0001

# set seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True

# training setting
epochs = 100
batch_size = 64
num_fold = 5
shuffle = True
#aug = True
aug = False

# cuda
cuda = 0
device = torch.device('cuda:'+ str(cuda))

if LOCAL:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

froze_encoder = False
normalize_rotation = False

model_name = 'CLS'  
latent_size = 1024
use_feature = ['z']

pos_weight = [1]

# dataset
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

ckpt_path = os.path.join(ckpt_folder, dataset_name, model_name, f'week{week}', "classification", model_ckpt)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

if LOCAL:
    ckpt_path = os.path.join(ckpt_folder, dataset_name, model_name, 'baseline_rig_eval_retrained')

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
    model = CLS(latent_size, use_feature=use_feature, dropout=(froze_encoder == False), gpu=None, num_conv=num_conv).to(device)
else:
    model = CLS(latent_size, use_feature=use_feature, dropout=(froze_encoder == False), gpu=device, num_conv=num_conv).to(device)

print("Model set!!!")

# froze encoder
if froze_encoder:
    for param in model.encoder.parameters():
        param.requires_grad = False

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
# optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5)

if phase != 'train':
    [model], start_epoch = load_checkpoint_by_key([model], ckpt_path, ['model'], device)

start_epoch = -1

total_params = sum(p.numel() for p in model.parameters()) # if p.requires_grad)
print(f"Total number of parameters in model: {total_params}")

# possible rotations
#rotations = [0, 90, 180, 270]
#rotations = [(0, 90), (90, 180), (180, 360)] # instead of fixed rotations, we will use random rotations
rotations = [(15, 45)] # train once with only one rotation
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
            "train_mode": "baseline for classification!",
            "week": week,
            "fold": fold,
            "architecture": model_name,
            "model_params": total_params,
            "learning_rate": lr,
            "data_subset": subj_list_postfix,
            "epochs": epochs,
            "batch_size": batch_size,
            "rotations": rotations,
            "froze_encoder": froze_encoder,
            "notes": f"downstream task, of week {week}, of {ckpt_path}"
        }
    )
    wandb.watch(model, log='all') # add gradient visualization

# ------- train -------
def train():
    print("Starting Training...")

    global_iter = 0
    monitor_metric_best = -1
    start_time = time.time()

    epoch_list = range(start_epoch + 1, epochs)
    # save losses for plotting
    train_cls_loss_list = []
    train_total_loss_list = []

    val_cls_loss_list = []
    val_total_loss_list = []
    
    # iterate through epochs
    for epoch in range(start_epoch + 1, epochs):
        model.train()
        loss_all_dict = {'all': 0, 'cls': 0.}
        global_iter0 = global_iter

        pred_list = [] 
        label_list = []

        print(f'Number of Training Batches: {len(trainDataLoader)}')

        # go through all data batches in dataset
        for iter, sample in enumerate(trainDataLoader, 0):
            for _ in range(1):
            #for rotation in rotations:
                #if type(rotation) != int:
                #    rotation = random.randint(rotation[0], rotation[1])
                img1 = sample['img'].to(device, dtype=torch.float)
                if _ == 0:
                    #rotation = random.choice(rotations) 
                    rotation = rotations[0]
                    if type(rotation) != int:
                        rotation = random.randint(rotation[0], rotation[1])
                    global_iter += 1
                    axis = axes[np.random.randint(0, len(axes) - 1)]
                    rot_mat = generate_rotation_matrix(axis, rotation, device)#.to(device, dtype=torch.float) # R
                    
                    img2 = rotate_batch(img1, batch_size, rot_mat)
                    img2 = img2.to(device, dtype=torch.float).unsqueeze(1)
                else:
                    img2 = img1.to(device, dtype=torch.float).unsqueeze(1)
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

                # run model on rotated image
                pred, _ = model.forward_single(img2)

                loss_cls, pred_sig = model.compute_classification_loss(pred, label, torch.tensor(pos_weight), dataset_name, subj_list_postfix)

                loss = loss_cls
                loss_all_dict['cls'] += loss_cls.item()
                loss_all_dict['all'] += loss.item()

                if DEBUG:
                    print(f'Sample classification loss: {loss_cls}')
                    print(f'Sample total loss: {loss_cls}')

                pred_list.append(pred_sig.detach().cpu().numpy())
                label_list.append(label.detach().cpu().numpy())

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
                    print('Epoch[%3d], iter[%3d]: loss=[%.4f], cls=[%.4f], rotation=[%.4f]' \
                            % (epoch, iter, loss.item(), loss_cls.item(), rotation))

        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter
        save_result_stat(loss_all_dict, {'ckpt_path': ckpt_path}, info='epoch[%2d]'%(epoch))
        print(loss_all_dict)
        train_cls_loss_list.append(loss_all_dict['cls'])
        train_total_loss_list.append(loss_all_dict['all'])

        pred_list = np.concatenate(pred_list, axis=0) # pred already sigmoid
        label_list = np.concatenate(label_list, axis=0)
        train_stat = compute_classification_metrics(label_list, pred_list, dataset_name, subj_list_postfix)

        # validation
        # pdb.set_trace()
        stat = evaluate(phase='val', set='val', save_res=False)
        monitor_metric = stat['bacc']
        scheduler.step(monitor_metric)
        save_result_stat(stat, {'ckpt_path': ckpt_path}, info='val')
        print(stat)
        val_cls_loss_list.append(stat['cls'])
        val_total_loss_list.append(stat['all'])

        wandb.log({
            "train_cls_loss": loss_all_dict['cls'][0],
            "validation_cls_loss": stat['cls'][0],
            "train_bacc": train_stat['bacc'],
            "validation_bacc": stat['bacc'][0],
            "train_f1": train_stat["f1"],
            "validation_f1": stat["f1"][0],
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
    print(zs_file_path)
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
    label_list = []
    recon1_list = []
    z1_list = []
    pred_list = []

    with torch.no_grad():
        for iter, sample in tqdm.tqdm(enumerate(loader, 0)):
            img1 = sample['img'].to(device, dtype=torch.float).unsqueeze(1)
            if subj_list_postfix == 'C_single':
                label = sample['age'].to(device, dtype=torch.float)
            else:
                label = sample['label'].to(device, dtype=torch.float)

            # run model
            pred, _ = model.forward_single(img1)

            loss_cls, pred_sig = model.compute_classification_loss(pred, label, torch.tensor(pos_weight), dataset_name, subj_list_postfix)

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
        stats = compute_classification_metrics(label_list, pred_list, dataset_name, subj_list_postfix)
        loss_all_dict['bacc'] = stats['bacc']
        loss_all_dict['f1'] = stats['f1']

        if phase == 'test' and save_res:
            img1_list = np.concatenate(img1_list, axis=0)
            h5_file = h5py.File(path, 'w')
            h5_file.create_dataset('img1', data=img1_list)
            h5_file.create_dataset('label', data=label_list)
            h5_file.create_dataset('pred', data=pred_list)

    if wandb_log:
        wandb.log({
            f"{phase}_loss": loss_all_dict['cls'], 
            f"{phase}_bacc": loss_all_dict['bacc'],
            f"{phase}_f1": loss_all_dict['f1']}
        )

    return loss_all_dict

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

if phase == 'train':
    train()
    [model], start_epoch = load_checkpoint_by_key([model], ckpt_path, ['model'], device)
    stat = evaluate(phase='test', set='test', save_res=True, wandb_log=True)
else:
    stat = evaluate(phase='test', set='test', save_res=True, info='batch')
    print(stat)
