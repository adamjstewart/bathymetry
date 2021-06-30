#!/usr/bin/env bash

#SBATCH -J spack
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=adamjs5@illinois.edu

rm -rf ~/bathymetry/.spack-env ~/bathymetry/spack.lock
spack --color=never env activate ~/bathymetry
spack --color=never install
