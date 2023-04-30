#!/bin/bash

#SBATCH -o myScript.sh.log-%j
#SBATCH --gres=gpu:volta:2
#SBATCH -n 9
#SBATCH -N 4

source /etc/profile
module load anaconda/2023a
module load mpi/openmpi-4.1.3
module load cuda/11.6

mpirun -np 9 --mca btl ^openib python main.py
