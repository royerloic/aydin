#!/usr/bin/env bash

conda create -n pitl python=3.7
conda activate pitl

conda install mkl numpy zarr scipy scikit-image scikit-learn click black dask pytest numexpr pyopencl lightgbm

pip install PIMS gdown tifffile czifile pre-commit black
pip install imagecodecs
