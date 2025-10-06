#!/usr/bin/env bash

set -euo pipefail

for alpha in 0 0.001 0.01 0.1 1 10 100 1000
do
    echo alpha: $alpha
    python3 $HOME/bathymetry/train.py ridge --alpha $alpha
done
