#!/usr/bin/env bash

#SBATCH --job-name=spack
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=small
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adamjs5@illinois.edu

rm -rf ~/bathymetry/.spack-env ~/bathymetry/spack.lock
spack --color=never env activate ~/bathymetry
spack --color=never install
