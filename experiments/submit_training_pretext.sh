#!/bin/bash
#
#SBATCH --job-name=model_training_pretext
#
#SBATCH --partition=kpohl
#SBATCH -G 1
#SBATCH --time=30:00:00
#SBATCH --mem=12G

/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/main_dist_loss_vn_module.py