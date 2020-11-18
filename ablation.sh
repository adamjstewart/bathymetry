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

for i in ${!features[*]}
do
    echo $i
    echo "${features[i]}"
    echo "${names[i]}"
    python train.py --ablation "${features[i]}" $model
    cp "checkpoints/$model.pickle" "checkpoints/${names[i]}-$model.pickle"
    python plot.py world $model truth
    cp "results/$model-truth.png" "results/${names[i]}-$model-truth.png"
done
