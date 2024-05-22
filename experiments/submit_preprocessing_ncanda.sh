#!/bin/bash
#
#SBATCH --job-name=data_preprocessing
#
#SBATCH --partition=kpohl
#SBATCH -G 1
#SBATCH --time=10:00:00
#SBATCH --mem=12G

/scratch/users/shizhehe/miniconda3/envs/mri_vnn/bin/python /home/users/shizhehe/mri_vnn/experiments/data_preprocessing_NCANDA.py