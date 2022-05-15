#!/usr/bin/env bash

python plot.py 2d

for model in hs psm gdh1 h13 linear ridge svr mlp
do
    python plot.py world $model
    python plot.py world $model truth
done

for ml in linear ridge svr mlp
do
    for physics in hs psm gdh1 h13
    do
        python plot.py world $ml $physics
    done
done
