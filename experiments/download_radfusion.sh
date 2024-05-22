#!/bin/bash
#
#SBATCH --job-name=data_preprocessing
#
#SBATCH --partition=kpohl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2G
#SBATCH --time=20:00:00

/scratch/users/shizhehe/azcopy_linux_amd64_10.24.0/azcopy copy 'https://aimistanforddatasets01.blob.core.windows.net/multimodalpulmonaryembolismdataset?sv=2019-02-02&sr=c&sig=JWa0Nfd%2FNloi2e6A8s6r%2B0RFCcKNOCESmq8agVwy9Aw%3D&st=2024-04-25T16%3A58%3A13Z&se=2024-05-25T17%3A03%3A13Z&sp=rl' '/scratch/groups/kpohl/temp_radfusion/' --recursive