#!/bin/bash
#SBATCH --job-name=training_gcn
#SBATCH --output=training_gcn.out
#SBATCH --error=training_gcn.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --partition=main
#SBATCH --mail-user=fangyunl@usc.edu
#SBATCH --mail-type=all

# Your commands here
module purge
eval "$(conda shell.bash hook)"
conda activate /home1/fangyunl/anaconda3/envs/CBB

python GCN_CNN_BiLSTM.py