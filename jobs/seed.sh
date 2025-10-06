#!/usr/bin/env bash

set -euo pipefail

model='mlp'

seeds=($(seq 1 10))

mkdir -p 'checkpoints/seed'
mkdir -p 'results/seed'

for s in ${seeds[@]}
do
    echo "seed: $s"
    python3 train.py --seed $s $model
    python3 plot.py world $model truth
    mv "checkpoints/$model.nc" "checkpoints/seed/$s-$model.nc"
    mv "results/residual/$model-truth.png" "results/seed/$s-$model-truth.png"
done

python3 plot.py seed
