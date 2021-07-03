#!/usr/bin/env bash

#SBATCH --job-name=svr
#SBATCH --time=2-00:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adamjs5@illinois.edu

spack --color=never env activate ~/bathymetry

# https://rcc.uchicago.edu/docs/running-jobs/srun-parallel/index.html#parallel-batch
ulimit -u 10000

srun="srun --exclusive -N1 -n1"
parallel="parallel --delay 0.2 -j $SLURM_NTASKS --joblog runtask-svr.log --resume-failed --verbose"
$parallel "$srun python $HOME/bathymetry/train.py svr --kernel {1} --gamma {2} --coef0 {3} --c {4} --epsilon {5}" ::: linear poly rbf sigmoid ::: scale auto ::: 0 0.0001 0.001 0.01 0.1 ::: 0.1 1 10 100 1000 ::: 0 0.01 0.1 1 10
