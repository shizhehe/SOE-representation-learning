#!/bin/bash
#
#SBATCH --job-name=model_training_downstream
#
#SBATCH --partition=kpohl
#SBATCH -G 1
#SBATCH --time=10:00:00
#SBATCH --mem=12G

/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/main_downstream_tl.py -fold 0 -name VN_Net_fold_{fold}_fixed_strong_max_lowlr_special_small
/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/main_downstream_tl.py -fold 1 -name VN_Net_fold_{fold}_fixed_strong_max_lowlr_special_small
/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/main_downstream_tl.py -fold 2 -name VN_Net_fold_{fold}_fixed_strong_max_lowlr_special_small
/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/main_downstream_tl.py -fold 3 -name VN_Net_fold_{fold}_fixed_strong_max_lowlr_special_small
/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/main_downstream_tl.py -fold 4 -name VN_Net_fold_{fold}_fixed_strong_max_lowlr_special_small