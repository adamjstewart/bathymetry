#!/usr/bin/env bash

#SBATCH --job-name=svr
#SBATCH --time=2-00:00:00
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adamjs5@illinois.edu

spack --color=never env activate ~/bathymetry
module load launcher

export LAUNCHER_WORKDIR=~/bathymetry
export LAUNCHER_JOB_FILE=svr-job-file.txt

${LAUNCHER_DIR}/paramrun
