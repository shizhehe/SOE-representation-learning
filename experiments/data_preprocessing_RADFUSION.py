
import os
import glob
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import scipy.ndimage
from datetime import datetime
import random
import pdb
import sys
from skimage.transform import resize

import medutils
import matplotlib.pyplot as plt

seed = 10
np.random.seed(seed)

num_slices = 128
reshape_to = (num_slices, 128, 128)


# preprocess subject label and data
#csv_path_raw = '/scratch/groups/kpohl/temp_radfusion/multimodalpulmonaryembolismdataset/Labels.csv'  # label for each timepoint (CN, AD, Demantia, MCI, LMCI, EMCI)
data_path = '/scratch/groups/kpohl/temp_radfusion/multimodalpulmonaryembolismdataset/'
#data_path = '/Users/shizhehe/dev/research/vector_neurons_mri/RADFUSION/'
#csv_path_raw = '/Users/shizhehe/dev/research/vector_neurons_mri/RADFUSION/Labels.csv'
csv_path_raw = '/scratch/groups/kpohl/temp_radfusion/multimodalpulmonaryembolismdataset/Labels.csv'
df_raw = pd.read_csv(csv_path_raw, usecols=['idx', 'label', 'pe_type', 'split'])

print(df_raw.head())

# load label, age, image paths
'''
struct subj_data

age: baseline age,
label: label for the subject, 0 - NC, 1 - PE
date_start: baseline date, in datetime format
date: list of dates, in datetime format
date_interval: list of intervals, in year
img_paths: list of image paths
'''

total_num = df_raw.shape[0]

img_paths = []
for i in range(0, 8):
    img_paths += glob.glob(data_path+f'{i}/*.npy')

img_paths = sorted(img_paths)
subj_data = {}
nan_label_count = 0
nan_idx_list = []
for img_path in img_paths:
    subj_id = int(os.path.splitext(os.path.basename(img_path))[0])
    rows = df_raw.loc[(df_raw['idx'] == subj_id)]
    if rows.shape[0] > 1:
        print(f'Multiple image for same patient with subj_id [{subj_id}]')
        print(rows)
    if rows.shape[0] == 0:
        print('Missing label for', subj_id)
    else:
        # build dict
        # only one sample per patient
        if subj_id not in subj_data:
            #subj_data[subj_id] = {'age': rows.iloc[0]['AGE'], 'label_all': [], 'label': label_dict[rows.iloc[i]['DX_bl']], 'date': [], 'date_start': date_struct, 'date_interval': [], 'img_paths': []}
            subj_data[str(subj_id)] = {
                #'age': rows.iloc[0]['AGE'], # currently not using age data for RADFUSION
                'label_all': [], 
                'label': rows.iloc[0]['label'], 
                'split': rows.iloc[0]['split'],
                'pe_type': rows.iloc[0]['pe_type'] if rows.iloc[0]['label'] == 1 else 'none',
                'img_paths': [img_path],
                'num_samples': rows.shape[0]
            }

# get sMCI, pMCI labels
num_ts_nc = 0
num_ts_pe = 0
num_nc = 0
num_pe = 0
subj_list_dict = {'NC':[], 'PE':[]}
for subj_id in subj_data.keys():
    if subj_data[subj_id]['label'] == 0: 
        num_nc += 1
        num_ts_nc += subj_data[subj_id]['num_samples']
        subj_list_dict['NC'].append(subj_id)
    elif subj_data[subj_id]['label'] == 1:
        num_pe += 1
        num_ts_pe += subj_data[subj_id]['num_samples']
        subj_list_dict['PE'].append(subj_id)

print('Number of timesteps, NC/PE:', num_ts_nc, num_ts_pe)
print('Number of subjects, NC/PE:', num_nc, num_pe)

print(subj_list_dict)

# save subj_list_dict to npy
np.save('/scratch/groups/kpohl/temp_radfusion/multimodalpulmonaryembolismdataset/RADFUSION/RADFUSION_longitudinal_subj.npy', subj_list_dict)
#np.save('/Users/shizhehe/dev/research/vector_neurons_mri/RADFUSION/RADFUSION_longitudinal_subj.npy', subj_list_dict)

# statistics about timesteps
"""max_timestep = 0
num_cls = [0,0,0,0,0]
num_ts = [0,0,0,0,0,0,0,0,0]
counts = np.zeros((5, 8))
for subj_id, info in subj_data.items():
    num_timestep = len(info['img_paths'])
    if len(info['label_all']) != num_timestep or len(info['date_interval']) != num_timestep:
        print('Different number of timepoint', subj_id)
    max_timestep = max(max_timestep, num_timestep)
    num_cls[info['label']] += 1
    num_ts[num_timestep] += 1
    counts[info['label'], num_timestep] += 1
print('Number of subjects: ', len(subj_data))
print('Max number of timesteps: ', max_timestep)
print('Number of each timestep', num_ts)
print('Number of each class', num_cls)
print('NC', counts[0])
print('sMCI', counts[3])
print('pMCI', counts[4])
print('AD', counts[2])

counts_cum = counts.copy()
for i in range(counts.shape[1]-2, 0, -1):
    counts_cum[:, i] += counts_cum[:, i+1]
print(counts_cum)"""

# save subj_data to h5
h5_noimg_path = '/scratch/groups/kpohl/temp_radfusion/multimodalpulmonaryembolismdataset/RADFUSION/RADFUSION_longitudinal_noimg.h5'
if not os.path.exists(h5_noimg_path):
    f_noimg = h5py.File(h5_noimg_path, 'a')
    for i, subj_id in enumerate(subj_data.keys()):
        subj_noimg = f_noimg.create_group(subj_id)
        subj_noimg.create_dataset('label', data=subj_data[subj_id]['label'])
        subj_noimg.create_dataset('label_all', data=subj_data[subj_id]['label_all'])
        subj_noimg.create_dataset('split', data=subj_data[subj_id]['split'])
        subj_noimg.create_dataset('pe_type', data=subj_data[subj_id]['pe_type'])
        subj_noimg.create_dataset('img_paths', data=subj_data[subj_id]['img_paths'])
        subj_noimg.create_dataset('num_samples', data=subj_data[subj_id]['num_samples'])
    f_noimg.close()

scan_sizes = []

subj_data_filtered = {}

sys.exit()

"""for i, subj_id in enumerate(subj_data.keys()):
    img_paths = subj_data[subj_id]['img_paths']
    print(img_paths)
    for img_path in img_paths:
        img = np.load(img_path)
        scan_sizes.append(img.shape)
        if img.shape[0] < num_slices:
            print(f'Too few slices, < {num_slices}, skipping {img_path}')
            continue
        else:
            subj_data_filtered[subj_id] = subj_data[subj_id]
    print(i, subj_id)

print(f'number of dropped: {len(subj_data) - len(subj_data_filtered)}')"""
    
# save images to h5
# statistics about scan size
h5_img_path = '/scratch/groups/kpohl/temp_radfusion/multimodalpulmonaryembolismdataset/RADFUSION/RADFUSION_longitudinal_img.h5'
#h5_img_path = '/Users/shizhehe/dev/research/vector_neurons_mri/RADFUSION/RADFUSION_longitudinal_img.h5'
to_few_count = 0
if not os.path.exists(h5_img_path):
    f_img = h5py.File(h5_img_path, 'a')
    for i, subj_id in enumerate(subj_data.keys()):
        subj_img = f_img.create_group(subj_id)
        img_paths = subj_data[subj_id]['img_paths']
        print(img_paths)
        for img_path in img_paths:
            img = np.load(img_path)
            scan_sizes.append(img.shape)
            if img.shape[0] < num_slices:
                print(f'Too few slices, < {num_slices}, skipping {img_path}')
                to_few_count += 1
                continue
            else:
                subj_data_filtered[subj_id] = subj_data[subj_id]
            img = resize(img, reshape_to, mode='constant', anti_aliasing=True)

            #img = (img - np.mean(img)) / np.std(img) image has already been normalized
            subj_img.create_dataset(os.path.basename(img_path), data=img)
        print(i, subj_id)
    f_img.close()

print(f'Too few slices, count: {to_few_count}')

subj_data = subj_data_filtered

first_dim = [scan_size[0] for scan_size in scan_sizes]
second_dim = [scan_size[1] for scan_size in scan_sizes]
print(f'number of unique first dim: {len(np.unique(first_dim))}')
print(f'max number of slices: {np.max(first_dim)}')
print(f'min number of slices: {np.min(first_dim)}')
print(f'mean number of slices: {np.mean(first_dim)}')
print(f'count for each slice number: {np.unique(first_dim, return_counts=True)}')


# plot and save histogram of first dim
plt.hist(first_dim, bins=50)
plt.xlabel('first dim')
plt.ylabel('count')
plt.title('Histogram of first dim')
#plt.savefig('/Users/shizhehe/dev/research/vector_neurons_mri/RADFUSION/hist_first_dim.png')
plt.savefig('/scratch/groups/kpohl/temp_radfusion/multimodalpulmonaryembolismdataset/RADFUSION/hist_first_dim.png')

if len(set(second_dim)) == 1:
    print(f'all images have the same width/height before downsampling: {second_dim[0]}')

def augment_image(img, rotate, shift, flip):
    # pdb.set_trace()
    img = scipy.ndimage.interpolation.rotate(img, rotate[0], axes=(1,0), reshape=False)
    img = scipy.ndimage.interpolation.rotate(img, rotate[1], axes=(0,2), reshape=False)
    img = scipy.ndimage.interpolation.rotate(img, rotate[2], axes=(1,2), reshape=False)
    img = scipy.ndimage.shift(img, shift[0])
    if flip[0] == 1:
        img = np.flip(img, 0) - np.zeros_like(img)
    return img

"""h5_img_path = '/scratch/users/shizhehe/RADFUSION/RADFUSION_longitudinal_img_aug.h5'
aug_size = 10
if not os.path.exists(h5_img_path):
    f_img = h5py.File(h5_img_path, 'a')
    for i, subj_id in enumerate(subj_data.keys()):
        subj_img = f_img.create_group(subj_id)
        img_paths = subj_data[subj_id]['img_paths']
        rotate_list = np.random.uniform(-2, 2, (aug_size-1, 3))
        shift_list =  np.random.uniform(-2, 2, (aug_size-1, 1))
        flip_list =  np.random.randint(0, 2, (aug_size-1, 1))
        for img_path in img_paths:
            img_nib = nib.load(os.path.join(data_path,img_path))
            img = img_nib.get_fdata()
            img = (img - np.mean(img)) / np.std(img)
            imgs = [img]
            for j in range(aug_size-1):
                imgs.append(augment_image(img, rotate_list[j], shift_list[j], flip_list[j]))
            imgs = np.stack(imgs, 0)
            subj_img.create_dataset(os.path.basename(img_path), data=imgs)
        print(i, subj_id)
    f_img.close()"""

def save_data_txt(path, subj_id_list, case_id_list):
    with open(path, 'w') as ft:
        for subj_id, case_id in zip(subj_id_list, case_id_list):
            ft.write(subj_id+' '+case_id+'\n')

# save txt, subj_id, case_id, case_number, case_id, case_number
def save_pair_data_txt(path, subj_id_list, case_id_list):
    with open(path, 'w') as ft:
        for subj_id, case_id in zip(subj_id_list, case_id_list):
            ft.write(subj_id+' '+case_id[0]+' '+case_id[1]+' '+str(case_id[2])+' '+str(case_id[3])+'\n')

def save_single_data_txt(path, subj_id_list, case_id_list):
    with open(path, 'w') as ft:
        for subj_id, case_id in zip(subj_id_list, case_id_list):
            ft.write(subj_id+' '+case_id[0]+' '+str(case_id[1])+'\n')

def get_subj_pair_case_id_list(subj_data, subj_id_list):
    subj_id_list_full = []
    case_id_list_full = []
    for subj_id in subj_id_list:
        case_id_list = subj_data[subj_id]['img_paths']
        for i in range(len(case_id_list)):
            for j in range(i+1, len(case_id_list)):
                subj_id_list_full.append(subj_id)
                case_id_list_full.append([case_id_list[i],case_id_list[j],i,j])

                # pdb.set_trace()
                # filter out pairs that are too close
                # if subj_data[subj_id]['date_interval'][j] - subj_data[subj_id]['date_interval'][i] >= 2:
                #     subj_id_list_full.append(subj_id)
                #     case_id_list_full.append([case_id_list[i],case_id_list[j],i,j])
    return subj_id_list_full, case_id_list_full

def get_subj_single_case_id_list(subj_data, subj_id_list):
    subj_id_list_full = []
    case_id_list_full = []
    for subj_id in subj_id_list:
        if subj_id not in subj_data:
            print(f'subj_id {subj_id} not in subj_data')
            continue
        case_id_list = subj_data[subj_id]['img_paths']
        for i in range(len(case_id_list)):
            subj_id_list_full.append(subj_id)
            case_id_list_full.append([case_id_list[i], i])
    return subj_id_list_full, case_id_list_full
 
#pdb.set_trace()
subj_list_postfix = 'NC_PE_single'
#subj_list_postfix = 'NC_AD_pMCI_sMCI_single'
# subj_list_postfix = 'NC_AD_pMCI_sMCI_far'
#subj_list_postfix = 'NC_AD_single'
#subj_list_postfix = 'NC_single'
# subj_list_postfix = 'pMCI_sMCI'
#subj_id_all = np.load('/Users/shizhehe/dev/research/vector_neurons_mri/ADNI/ADNI_longitudinal_subj.npy', allow_pickle=True).item()
subj_id_all = np.load('/scratch/groups/kpohl/temp_radfusion/multimodalpulmonaryembolismdataset/RADFUSION/RADFUSION_longitudinal_subj.npy', allow_pickle=True).item()
#subj_id_all = np.load('/Users/shizhehe/dev/research/vector_neurons_mri/RADFUSION/RADFUSION_longitudinal_subj.npy', allow_pickle=True).item()
subj_id_all['NC'] = [subj_id for subj_id in subj_id_all['NC'] if subj_id in subj_data]
subj_id_all['PE'] = [subj_id for subj_id in subj_id_all['PE'] if subj_id in subj_data]

subj_list = []

print(subj_id_all)

print(len(subj_id_all['NC']))
print(len(subj_id_all['PE']))

for fold in range(5):
    subj_test_list = []
    subj_val_list = []
    subj_train_list = []

    for class_name in ['NC', 'PE']:
        class_list = subj_id_all[class_name]
        np.random.shuffle(class_list)
        num_class = len(class_list)

        class_test = [item for item in class_list if subj_data[item]['split'] == 'test']
        class_val = [item for item in class_list if subj_data[item]['split'] == 'val']
        class_train = [item for item in class_list if subj_data[item]['split'] == 'train']

        subj_test_list.extend(class_test)
        subj_train_list.extend(class_train)
        subj_val_list.extend(class_val)

    subj_id_list_train, case_id_list_train = get_subj_single_case_id_list(subj_data, subj_train_list)
    subj_id_list_val, case_id_list_val = get_subj_single_case_id_list(subj_data, subj_val_list)
    subj_id_list_test, case_id_list_test = get_subj_single_case_id_list(subj_data, subj_test_list)

    print(subj_id_list_train)
    print(case_id_list_train)

    path = '/scratch/groups/kpohl/temp_radfusion/multimodalpulmonaryembolismdataset/RADFUSION/fold'
    #path = '/Users/shizhehe/dev/research/vector_neurons_mri/RADFUSION/fold'
    save_single_data_txt(path+str(fold)+'_train_' + subj_list_postfix + '.txt', subj_id_list_train, case_id_list_train)
    save_single_data_txt(path+str(fold)+'_val_' + subj_list_postfix + '.txt', subj_id_list_val, case_id_list_val)
    save_single_data_txt(path+str(fold)+'_test_' + subj_list_postfix + '.txt', subj_id_list_test, case_id_list_test)
