#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --output=testing.out
#SBATCH --error=testing.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=5:00:00
#SBATCH --partition=main
#SBATCH --mail-user=fangyunl@usc.edu
#SBATCH --mail-type=all

# Your commands here
module purge
eval "$(conda shell.bash hook)"
conda activate CBB

python testing.py