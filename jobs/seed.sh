#!/usr/bin/env bash

set -euo pipefail

model='mlp'

seeds=($(seq 1 10))
grids=(
    1
    2
    3
    4
    5
    6
    9
    10
    12
    15
    18
    20
    30
    36
    45
    60
    90
)

mkdir -p 'checkpoints/seed'
mkdir -p 'results/seed'

for g in ${grids[@]}
do
    for s in ${seeds[@]}
    do
        echo "Grid size: $g, seed: $s"
        python3 train.py --grid-size $g --seed $s $model
        python3 plot.py world $model truth
        mv "checkpoints/$model.nc" "checkpoints/seed/$g-$s-$model.nc"
        mv "results/residual/$model-truth.png" "results/seed/$g-$s-$model-truth.png"
    done

    python3 plot.py seed -g $g
done
