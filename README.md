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

To install `fastbox`, simply run `python setup.py install`. The following are required dependencies (all of which can be installed via `pip`):

* `numpy>=1.18`
* `scipy>=1.5`
* `matplotlib>=2.2`
* `scikit-learn`
* `pyccl`

The following optional dependencies are needed for some of the foreground modelling and filtering functions to work:

* `healpy`
* `lmfit`
* `multiprocessing`

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
 - Calculate power spectra, correlation functions, and their 
   multipoles, via `nbodykit`

