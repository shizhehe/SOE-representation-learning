#!/bin/bash
#
#SBATCH --job-name=model_training_downstream
#
#SBATCH --partition=kpohl
#SBATCH -G 1
#SBATCH --time=01:00:00
#SBATCH --mem=12G

/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/evaluate_robustness_age.py -fold 0
/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/evaluate_robustness_age.py -fold 1
/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/evaluate_robustness_age.py -fold 2
/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/evaluate_robustness_age.py -fold 3
/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/evaluate_robustness_age.py -fold 4