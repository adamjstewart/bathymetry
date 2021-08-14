#!/usr/bin/env python3

import glob

# checkpoints/checkpoint-mlp-tanh-1.0-200-0.9-0.999-False-1e-08-[512]-constant-0.01-15000-200-0.9-10-True-0.5-None-True-sgd-0.0001-0.1-True-False.pickle

# checkpoints/checkpoint-mlp-tanh-1.0-200-0.9-0.999-False-1e-08-[512, 512, 512]-constant-1e-06-15000-200-0.9-10-True-0.5-None-True-lbfgs-0.0001-0.1-True-False.pickle

matches = []
for f in glob.iglob('checkpoints/checkpoint-mlp-*'):
    parts = f.split('-')
    activation = parts[2]
    alpha = float(parts[3])
    hidden_layer_sizes = parts[10].lstrip('[').rstrip(']').split(', ')
    hidden_layers = len(hidden_layer_sizes) + 2
    hidden_size = int(hidden_layer_sizes[0])
    learning_rate = float('-'.join(parts[12:-13]))
    solver = parts[-5]
    match = (activation, solver, alpha, hidden_layers, hidden_size, learning_rate)
    matches.append(match)

# python /home1/08020/tg873195/bathymetry/train.py mlp --activation logistic --solver lbfgs --alpha 0.001 --hidden-layers 3 --hidden-size 128 --learning-rate 0.01

with open('mlp-job-file.txt') as fo:
    with open('mlp-job-file.txt-new', 'w') as fn:
        for line in fo:
            parts = line.split()
            activation = parts[4]
            solver = parts[6]
            alpha = float(parts[8])
            hidden_layers = int(parts[10])
            hidden_size = int(parts[12])
            learning_rate = float(parts[14])
            match = (activation, solver, alpha, hidden_layers, hidden_size, learning_rate)
            if match not in matches:
                fn.write(line)
