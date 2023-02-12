#!/usr/bin/env bash

# for year in 2020 2019 2016 2013 2008
# do
#     # Plate
#     python plot.py -y $year 2d
# done

for model in hs psm gdh1 h13 linear ridge svr mlp
do
    # Bathymetry
    python plot.py world $model

    # Residual
    python plot.py world $model truth
done

for ml in linear ridge svr mlp
do
    for plate in hs psm gdh1 h13
    do
        # Residual
        python plot.py world $ml $plate
    done
done

# Features
python plot.py feature age age

for layer in water ice sediments crust moho
do
    for feature in thickness p s density
    do
        if [[ $layer == moho && $feature == thickness ]]
        then
            continue
        fi

        # Features
        python plot.py feature $layer $feature
    done
done
