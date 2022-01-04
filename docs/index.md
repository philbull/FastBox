# FastBox

Fast simulations of cosmological density fields, subject to anisotropic filtering, biasing, redshift-space distortions, foregrounds etc. This is intended to be a fast and simple simulator for post-EoR 21cm intensity maps and their cross-correlation with galaxy samples, with enough complexity to test realistic cosmological analysis methods.


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

If you would like to use COLA to generate fast, approximate N-body simulations (see the 
[``CosmoBox.realise_density_cola()``](box.md) method), please install [https://github.com/philbull/pycola3](pycola3).

In many of the examples, we also use `nbodykit` to analyse the simulations (e.g. to produce power spectra and 
correlation functions). Note that `nbodykit` may have package dependency version clashes with `pyccl` for 
some versions of Python, so you may need to install one of them manually (from source).

See the `examples/` directory for some example Jupyter notebooks and scripts.

## Current features

* Gaussian and log-normal density fields for any cosmology
* COLA approximate N-body simulated density fields
* Redshift-space transform, linear biasing etc
* Arbitrary anisotropic filters as a function of kperp and kparallel
* Poisson realisations of halo/galaxy samples
* Radiometer noise and beam convolutions (FFT and direct convolution)
* Several diffuse and point source foreground models, including GSM, Planck 
  Sky Model, and the Battye et al. point source model.
* Foreground filtering via PCA, ICA, Kernel PCA, least-squares etc.
* Integration with the DESC Core Cosmology Library (`pyccl`)
* Power spectra, correlation functions, and their multipoles, via `nbodykit`
