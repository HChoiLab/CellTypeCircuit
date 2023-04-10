#!/bin/bash
#SBATCH -Jlb10ns1.0e0.2_60
#SBATCH --account=gts-hchoi387
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --mem-per-cpu=16G
#SBATCH -t4:00:00
#SBATCH -qembers
#SBATCH -oreports/lb10ns1.0e0.2-60-%j.out
cd /storage/home/hcoda1/5/skim3438/scratch/CellTypeCircuit/newCode/data/lb10ns1.0e0.2
module load anaconda3
conda activate nest3
python ../../code/run_single_v2.py 1 7
