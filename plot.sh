#!/usr/bin/env bash

python plot.py 2d

for model in psm gdh1 h13 linear svr mlp
do
    python plot.py world $model
    python plot.py world $model truth
done

for ml in linear svr mlp
do
    for physics in psm gdh1 h13
    do
        python plot.py world $ml $physics
    done
done
