#!/bin/bash
#
#SBATCH --job-name=data_preprocessing
#
#SBATCH --time=200:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G

/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/data_preprocessing_ADNI.py