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
#SBATCH -o _logs_slurm/train/%x_%j.out
#SBATCH -e _logs_slurm/train/%x_%j.err


source 