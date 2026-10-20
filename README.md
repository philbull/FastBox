# FastBox

[![Documentation Status](https://readthedocs.org/projects/fastbox/badge/?version=latest)](https://fastbox.readthedocs.io/en/latest/?badge=latest)

Fast simulations of cosmological density fields, 
subject to anisotropic filtering, biasing, 
redshift-space distortions, foregrounds etc. This 
is intended to be a fast and simple simulator for 
post-EoR 21cm intensity maps and their cross-correlation 
with galaxy samples, with enough complexity to test 
realistic cosmological analysis methods.

## Installation

`fastbox` needs Python 3.10 or later. To install it, run `pip install .` from the repository directory. The following are required dependencies, which `pip` installs automatically:

* `numpy>=1.18`
* `scipy>=1.5`
* `matplotlib>=2.2`
* `scikit-learn`
* `scikit-image`
* `tqdm`
* `pyccl`

The following optional dependencies are needed for some of the foreground modelling and filtering functions to work:

* `healpy`
* `lmfit`
* `multiprocessing`
* `GPy`

## INSTALLATION IN A CONDA ENVIRONMENT

```bash
conda create -n fast-env python=3.12 pyccl scikit-learn scikit-image healpy lmfit -c conda-forge -y

git clone https://github.com/philbull/FastBox.git

cd FastBox

pip install .
```

If you want to use katbeam model:

```bash
pip install git+https://github.com/ska-sa/katbeam.git
```

## Current features

 - Gaussian and log-normal density fields for any cosmology
 - Redshift-space transform, linear biasing etc
 - Arbitrary anisotropic filters as a function of 
   kperp and kparallel
 - Poisson realisations of halo/galaxy samples
 - Radiometer noise and beam convolutions (FFT and direct convolution)
 - Several diffuse and point source foreground models, including GSM, Planck 
   Sky Model, and the Battye et al. point source model.
 - Foreground filtering via PCA, ICA, Kernel PCA, least-squares etc.
 - Integration with the DESC Core Cosmology Library (`pyccl`)
 - Calculate power spectra, cross spectra, correlation functions and 
   bispectra, and forward-model them on the simulation grid (`fastbox.power`)
 - Detect and generate catalogues of cosmic voids

