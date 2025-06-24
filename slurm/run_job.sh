#!/bin/bash
#SBATCH --job-name=vehicle_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# Activate your conda environment


# Run YOLOv10m training
python train.py