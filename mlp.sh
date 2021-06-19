#!/usr/bin/env bash

#SBATCH --account=geol-ljliu-group
#SBATCH --job-name=mlp
#SBATCH --mail-type=ALL
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --partition=ljliu
#SBATCH --time=6-00:00:00

spack --color=never env activate ~/bathymetry

# https://rcc.uchicago.edu/docs/running-jobs/srun-parallel/index.html#parallel-batch
ulimit -u 10000

srun="srun --exclusive -N1 -n1"
parallel="parallel --delay 0.2 -j $SLURM_NTASKS --joblog runtask-mlp.log --resume-failed --verbose"
$parallel "$srun python $HOME/bathymetry/train.py mlp --activation {1} --solver {2} --alpha {3} --hidden-layers {4} --hidden-size {5} --learning-rate {6}" ::: logistic tanh relu ::: lbfgs sgd adam ::: 0.001 0.01 0.1 1 10 ::: 3 4 5 6 7 ::: 128 256 512 1024 2056 ::: 0.01 0.001 0.0001 0.00001 0.000001
