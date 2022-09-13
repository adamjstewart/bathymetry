#!/usr/bin/env bash

model='mlp'

features=(
    'thickness'
    'p-wave velocity'
    's-wave velocity'
    'density'
    'age'
    'p-wave velocity,s-wave velocity,age'

    'water'
    'ice'
    'upper sediments,middle sediments,lower sediments'
    'upper crystalline crust,middle crystalline crust,lower crystalline crust'
    'moho'
)

names=(
    'no-thickness'
    'no-vp'
    'no-vs'
    'no-density'
    'no-age'
    'only-isostatic'

    'no-water'
    'no-ice'
    'no-sediments'
    'no-crust'
    'no-moho'
)

mkdir -p 'checkpoints/ablation'
mkdir -p 'results/ablation'

for i in ${!features[*]}
do
    echo $i
    echo "${features[i]}"
    echo "${names[i]}"
    python3 train.py --ablation "${features[i]}" $model
    python3 plot.py world $model truth
    mv "checkpoints/$model.nc" "checkpoints/ablation/${names[i]}-$model.nc"
    mv "results/$model-truth.png" "results/ablation/${names[i]}-$model-truth.png"
done
