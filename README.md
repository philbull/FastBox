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

## Differentiable Gaussian-box pipeline

Install the optional JAX extra with `pip install '.[jax]'`. Prepare the fixed
box geometry and pyCCL inputs outside `jax.jit` and `jax.grad`. The kernels
accept numerical inputs, so a differentiable cosmology can also supply the
power grid and the background values in `RSDInputs`.

```python
import jax
import jax.numpy as jnp
from fastbox.box import CosmoBox, default_cosmo
from fastbox import jax_backend as fbj

box = CosmoBox(default_cosmo, redshift=0.8, nsamp=16, realise_now=False)
pk = box.matter_power_grid(linear=True)
rsd = fbj.prepare_rsd(box)
bins = fbj.prepare_power_bins(box)
ang_x, ang_y = box.pixel_array()
white_noise = jax.random.normal(jax.random.key(0), box.k.shape)

def statistic(bias, sigma_v, beam_width):
    field = fbj.gaussian_density(white_noise, pk, box.boxfactor)
    field = fbj.linear_rsd_density(field, rsd, bias=bias, sigma_v=sigma_v)
    field = fbj.convolve_gaussian_beam(field, ang_x, ang_y, beam_width)
    _, power, _ = fbj.estimate_power_spectrum(field, bins, box.boxfactor)
    return jnp.nansum(power)

gradient = jax.jit(jax.grad(statistic, argnums=(0, 1, 2)))
d_bias, d_sigma_v, d_beam = gradient(1.6, 120.0, 1.0)
```

Keep the white-noise draw fixed when computing pathwise gradients. Use
`JAX_ENABLE_X64=1` when comparing against the NumPy implementation in double
precision. The catalogue, interpolated RSD, and foreground-cleaning routines
use the existing NumPy/SciPy paths.

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

