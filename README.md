# bathymetry
:ocean: Machine learning model for predicting ocean bathymetry

[![black](https://github.com/adamjstewart/bathymetry/actions/workflows/black.yaml/badge.svg)](https://github.com/adamjstewart/bathymetry/actions/workflows/black.yaml)
[![flake8](https://github.com/adamjstewart/bathymetry/actions/workflows/flake8.yaml/badge.svg)](https://github.com/adamjstewart/bathymetry/actions/workflows/flake8.yaml)
[![isort](https://github.com/adamjstewart/bathymetry/actions/workflows/isort.yaml/badge.svg)](https://github.com/adamjstewart/bathymetry/actions/workflows/isort.yaml)
[![mypy](https://github.com/adamjstewart/bathymetry/actions/workflows/mypy.yaml/badge.svg)](https://github.com/adamjstewart/bathymetry/actions/workflows/mypy.yaml)

## System Requirements

The versions listed below are what was used in our paper. Newer or older versions may also work. If you encounter any issues with newer versions, please open an issue.

### Software Dependencies

* Python 3.11.7
* cartopy 0.22.0
* cmocean 3.0.3
* geocube 0.4.2
* geopandas 0.14.1
* matplotlib 3.8.2
* netcdf4 1.6.5
* numpy 1.26.2
* pandas 2.1.4
* scikit-learn 1.3.2
* scipy 1.11.4
* shapely 2.0.2
* xarray 2023.12.0

### Operating Systems

* macOS 14.1.2
* Ubuntu 22.04.3

### Hardware Requirements

Should run on any CPU or RAM size, including on a laptop

## Installation Guide

First, clone this project:
```console
> git clone https://github.com/adamjstewart/bathymetry.git
> cd bathymetry
```
Then, install the Python dependencies:
```console
> pip install -r requirements.txt
```
This should only take a few seconds to install.

## Data

All data should be stored in the same root directory. The default is `data`, but a different directory can be specified with `--data-dir`.

### CRUST1.0

This model is trained on the [CRUST1.0](https://igppweb.ucsd.edu/~gabi/crust1.html) dataset. In order to reproduce this work, you will need to download both the [basic model](http://igppweb.ucsd.edu/~gabi/crust1/crust1.0.tar.gz) and the [add-on](http://igppweb.ucsd.edu/~gabi/crust1/crust1.0-addon.tar.gz) that includes the crustal type file. Then, extract the tarballs in a `crust1.0` directory within the data directory.

### Seafloor Age

Seafloor age data can be found at [EarthByte](https://www.earthbyte.org/category/resources/data-models/seafloor-age/). For this model, we downsample all seafloor age data to 1-degree resolution. We test with several different seafloor age datasets:

* [age2020](https://www.earthbyte.org/webdav/ftp/earthbyte/agegrid/2020/Grids/age.2020.1.GTS2012.6m.nc)
* [age2019](https://www.earthbyte.org/webdav/ftp/Data_Collections/Muller_etal_2019_Tectonics/Muller_etal_2019_Agegrids/Muller_etal_2019_Tectonics_v2.0_PresentDay_AgeGrid.nc)
* [age2016](https://www.earthbyte.org/webdav/ftp/Data_Collections/Muller_etal_2016_AREPS/Muller_etal_2016_AREPS_Agegrids/Muller_etal_2016_AREPS_Agegrids_v1.17/Muller_etal_2016_AREPS_v1.17_PresentDay_AgeGrid.nc)
* [age2013](https://www.earthbyte.org/webdav/ftp/papers/Muller_etal_OceanChemistry/Grids/agegrid_0.nc)
* [age2008](https://www.earthbyte.org/webdav/ftp/Data_Collections/Muller_etal_2008_G3/Seafloor_ages/age.3.6.unscaled.nc)

Each of these files should be placed in their respective directories within the data directory.

### Plate Boundaries

The plate boundaries shapefiles can be downloaded from the [World tectonic plates and boundaries](https://github.com/fraxen/tectonicplates). Download and extract a zip file of the entire repository within the data directory.

## Demo

To train a ridge regression model, run the following command:
```console
> python3 train.py ridge

Reading datasets...
Reading data/age2020/age.2020.1.GTS2012.6m.nc...
Reading data/crust1.0/crust1.bnds...
Reading data/crust1.0/crust1.vp...
Reading data/crust1.0/crust1.vs...
Reading data/crust1.0/crust1.rho...
Reading data/crust1.0/CNtype1-1.txt...
Reading data/tectonicplates-master/PB2002_plates.shp...

Preprocessing...

Cross-validation...
Group 1
Group 2
Group 3
Group 4
Group 5
Group 6
Group 7

Evaluating...
RMSE: 0.591818050597389
R^2:  0.7508725821083216

Saving predictions...
Writing checkpoints/checkpoint-ridge-100-True-False-None-False-1-auto-0.0001.pickle...
Writing checkpoints/truth.nc...
Writing checkpoints/ridge.nc...
```
This should only take a few seconds to run. Replace "ridge" with other models to compare performance metrics. Note that MLP will take much longer (around an hour on a laptop).

## Reproducibility

To reproduce all experimental results from our paper, see the scripts in the `jobs` directory. Specifically:

* `ridge*.sh`, `svr*.sh`, `mlp*.sh`: find optimal hyperparameters for all models
* `train.sh`: reproduce results with optimal hyperparameters
* `ablation.sh`: feature and layer ablation study
* `plot.sh`: generate some basic maps of the results

These jobs were submitted using the Slurm Workload Manager on TACC and ICCP. The scripts should work on any system, but may be slow unless you use a cluster. If you use a different cluster, you may need to change the job configuration parameters.
