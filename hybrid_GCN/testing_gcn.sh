#!/bin/bash
#SBATCH --job-name=testing_gcn
#SBATCH --output=testing_gcn.out
#SBATCH --error=testing_gcn.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --mail-user=fangyunl@usc.edu
#SBATCH --mail-type=all

# Your commands here
module purge
eval "$(conda shell.bash hook)"
conda activate /home1/fangyunl/anaconda3/envs/CBB

cd /scratch1/fangyunl/DeepVirData/GCN

#for testing
python testing_gcn.py