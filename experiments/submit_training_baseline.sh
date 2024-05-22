#!/bin/bash
#
#SBATCH --job-name=model_training
#
#SBATCH --partition=kpohl
#SBATCH -G 1
#SBATCH --time=20:00:00
#SBATCH --mem-per-gpu=32G

/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/main_baseline.py -fold 0
/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/main_baseline.py -fold 1
/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/main_baseline.py -fold 2
/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/main_baseline.py -fold 3
/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/main_baseline.py -fold 4