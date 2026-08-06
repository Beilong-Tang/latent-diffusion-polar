#!/bin/bash
#SBATCH -J train_latent_diffusion
#SBATCH -p gpu-hp
#SBATCH --qos=ncsu_h200_hp
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 96:00:00
#SBATCH -o _logs_slurm/%x_%j.out
#SBATCH -e _logs_slurm/%x_%j.err
set -e 

# Initialization
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /work/btang1/envs/ldm

CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml -t --gpus 0,  --resume_latest --load_cuda_data