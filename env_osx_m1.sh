#  To install on a M1 machine, execute the following one liner:
#  conda deactivate; conda env remove -y -n aydin; conda create -y -n aydin python=3.9; conda activate aydin; chmod +x env_osx_m1.sh; ./env_osx_m1.sh
pip install -e . --no-dependencies
conda install -y -c conda-forge   Click==8.0.3
conda install -y -c conda-forge   czifile #==2019.7.2
conda install -y -c conda-forge   gdown #==4.2.0
conda install -y -c conda-forge   googledrivedownloader #==0.4
conda install -y -c conda-forge   importlib-metadata #==4.10.0
conda install -y -c conda-forge   jsonpickle #==1.3.0
conda install -y -c conda-forge   lightgbm #==3.3.1
conda install -y -c conda-forge   nd2reader #==3.3.0
conda install -y -c conda-forge   numba #==0.54.1
conda install -y -c conda-forge   numexpr #==2.7.3
# conda install -y -c conda-forge   numpy #>=1.19.2
conda install -y -c conda-forge   pynndescent #==0.5.5
conda install -y -c conda-forge   pyqt
conda install -y -c conda-forge   QDarkStyle #==3.0.2
conda install -y -c conda-forge   qtpy #==1.11.2
conda install -y -c conda-forge   scikit-image #==0.18.3
conda install -y -c conda-forge   scipy==1.7.3
conda install -y -c conda-forge   tensorflow #==2.7.0
conda install -y -c conda-forge   pytorch #==1.10.1
conda install -y -c conda-forge   keras #==2.7.0
conda install -y -c conda-forge   zarr #==2.4.0
conda install -y -c conda-forge   docstring-parser==0.13
conda install -y -c conda-forge   napari #==0.4.12
