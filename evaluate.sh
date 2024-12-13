#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --output=evaluate.out
#SBATCH --error=evaluate.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --mail-user=fangyunl@usc.edu
#SBATCH --mail-type=all

# Your commands here
module purge
eval "$(conda shell.bash hook)"
conda activate /home1/fangyunl/anaconda3/envs/CBB

cd /scratch1/fangyunl/DeepVirData

#for testing
python evaluate.py