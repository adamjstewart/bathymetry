# bathymetry
:ocean: Machine learning model for predicting ocean bathymetry

[![mypy](https://github.com/adamjstewart/bathymetry/workflows/mypy/badge.svg)](https://github.com/adamjstewart/bathymetry/actions)
[![flake8](https://github.com/adamjstewart/bathymetry/workflows/flake8/badge.svg)](https://github.com/adamjstewart/bathymetry/actions)
[![black](https://github.com/adamjstewart/bathymetry/workflows/black/badge.svg)](https://github.com/adamjstewart/bathymetry/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

First, clone this project:
```console
$ git clone https://github.com/adamjstewart/bathymetry.git
$ cd bathymetry
```
Then, install the Python dependencies. This can be done with Spack:
```console
$ spack env activate .
$ spack install
```
or with pip:
```console
$ pip install -r requirements.txt
```

## Data

### CRUST 1.0

This model is trained on the [CRUST 1.0](https://igppweb.ucsd.edu/~gabi/crust1.html) dataset. In order to reproduce this work, you will need to download both the [basic model](http://igppweb.ucsd.edu/~gabi/crust1/crust1.0.tar.gz) and the [add-on](http://igppweb.ucsd.edu/~gabi/crust1/crust1.0-addon.tar.gz) that includes the crustal type file. Then, extract the tarballs in a `data/crust1.0` folder, or specify a different directory with `--data-dir` when you run the model.

### Seafloor Age

Seafloor age data can be found at [EarthByte](https://www.earthbyte.org/category/resources/data-models/seafloor-age/). For this model, we downsample all seafloor age data to 1-degree resolution. We test with several different seafloor age datasets:

* [age2020](https://www.earthbyte.org/webdav/ftp/earthbyte/agegrid/2020/Grids/age.2020.1.GTS2012.6m.nc)

Each of these files should be placed in their respective directories.

### Plate Boundaries

The plate boundaries shapefiles can be downloaded from the [World tectonic plates and boundaries](https://github.com/fraxen/tectonicplates). Download and extract a zip file of the entire repository.
