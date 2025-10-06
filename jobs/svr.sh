#!/usr/bin/env bash

set -euo pipefail

for kernel in linear poly rbf sigmoid
do
    for gamma in scale auto
    do
        for coef0 in 0 0.0001 0.001 0.01 0.1
        do
            for c in 0.1 1 10 100 1000
            do
                for epsilon in 0 0.01 0.1 1 10
                do
                    echo kernel: $kernel, gamma: $gamma, coef0: $coef0, C: $c, epsilon: $epsilon
                    python3 $HOME/bathymetry/train.py svr --kernel $kernel --gamma $gamma --coef0 $coef0 --c $c --epsilon $epsilon
                done
            done
        done
    done
done
