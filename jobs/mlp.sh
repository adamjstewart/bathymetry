#!/usr/bin/env bash

set -euo pipefail

for activation in logistic tanh relu
do
    for solver in lbfgs sgd adam
    do
        for alpha in 0.001 0.01 0.1 1 10
        do
            for hidden_layers in 3 4 5 6 7
            do
                for hidden_size in 128 256 512 1024 2056
                do
                    for learning_rate in 0.01 0.001 0.0001 0.00001 0.000001
                    do
                        echo activation: $activation, solver: $solver, alpha: $alpha, layers: $hidden_layers, size: $hidden_size, lr: $learning_rate
                        python3 $HOME/bathymetry/train.py mlp --activation $activation --solver $solver --alpha $alpha --hidden-layers $hidden_layers --hidden-size $hidden_size --learning-rate $learning_rate
                    done
                done
            done
        done
    done
done
