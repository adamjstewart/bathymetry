#!/usr/bin/env bash

for model in hs psm gdh1 h13 linear svr mlp
do
    python train.py $model
done
