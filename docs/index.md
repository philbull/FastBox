# FastBox

Fast simulations of cosmological density fields, subject to anisotropic filtering, biasing, redshift-space distortions, foregrounds etc. This is intended to be a fast and simple simulator for post-EoR 21cm intensity maps and their cross-correlation with galaxy samples, with enough complexity to test realistic cosmological analysis methods.


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

If you would like to use COLA to generate fast, approximate N-body simulations (see the 
[``CosmoBox.realise_density_cola()``](box.md) method), please install [pycola3](https://github.com/philbull/pycola3).

Power spectra, cross spectra, correlation functions and bispectra of the simulations can be measured with
`fastbox.power`, which also forward-models the theory onto the simulation grid (see [Power spectra](power.md)).

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
* In-painting RFI gaps using Wiener filtering and LSSA
