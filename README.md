# bathymetry
:ocean: Machine learning model for predicting ocean bathymetry

## Installation

First, clone this project:
```console
$ git clone https://github.com/adamjstewart/bathymetry.git
$ cd bathymetry
```
Then, install the Python dependencies:
```console
$ spack install  # if you prefer Spack
$ pip install requirements.txt  # if you prefer pip
```

## Data

This model is trained on the [CRUST 1.0](https://igppweb.ucsd.edu/~gabi/crust1.html) dataset. In order to reproduce this work, you will need to download both the [basic model](http://igppweb.ucsd.edu/~gabi/crust1/crust1.0.tar.gz) and the [add-on](http://igppweb.ucsd.edu/~gabi/crust1/crust1.0-addon.tar.gz) that includes the crustal type file. Then, extract the tarballs in a `data/CRUST1.0` folder, or specify a different directory with `--data-dir` when you run the model.
