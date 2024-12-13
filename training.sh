#!/bin/bash
#SBATCH --job-name=training
#SBATCH --output=training.out
#SBATCH --error=training.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --partition=main
#SBATCH --mail-user=fangyunl@usc.edu
#SBATCH --mail-type=all

# Your commands here
module purge
eval "$(conda shell.bash hook)"
conda activate /home1/fangyunl/anaconda3/envs/CBB

python model_training.py