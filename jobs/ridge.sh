#!/usr/bin/env bash

# No need for slurm, training is very fast

for alpha in 0 0.001 0.01 0.1 1 10 100 1000
do
    python $HOME/bathymetry/train.py ridge --alpha $alpha
done
