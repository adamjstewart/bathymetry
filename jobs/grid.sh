#!/usr/bin/env bash

set -euo pipefail

model='mlp'

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

mkdir -p 'checkpoints/grid'
mkdir -p 'results/grid'

for g in ${grids[@]}
do
    echo "Grid size: $g"
    python3 train.py --grid-size $g $model
    python3 plot.py world $model truth
    mv "checkpoints/$model.nc" "checkpoints/grid/$g-$model.nc"
    mv "results/residual/$model-truth.png" "results/grid/$g-$model-truth.png"
done
