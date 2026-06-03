---
title: 'FastBox: A Lightweight Python Package for Fast Cosmological Signal Simulations'
tags:
  - Python
  - cosmology
  - 21cm intensity mapping
  - signal simulations
  - foreground removal
authors:
  - name: [YOUR NAME]
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: [YOUR INSTITUTION]
    index: 1
    ror: [YOUR ROR ID]
date: 02 June 2026
bibliography: paper.bib
---

# Summary

`FastBox` is a Python package for generating fast, physically realistic simulations of cosmological signals in three-dimensional co-moving boxes, with the primary focus of its application being 21cm intensity mapping (IM) experiments. It provides a framework for producing cosmology-dependent Gaussian and log-normal density fields, as well as modelling the effects of redshift-space distortions and linear biasing, among others. Models of instrumental systematics are incorporated, including radiometer noise and beam convolutions. Diffuse and point source foreground models are included, along with a number of foreground filtering strategies such as PCA, ICA, and transfer function correction via mock signal injection. Lastly, implementations for the calculation of power spectra and correlation functions. `FastBox` is designed as a lightweight but realistic test-bench for the development and validation of end-to-end cosmological analysis pipelines.

# Statement of Need

Probes of the 21cm, or neutral hydrogen (HI), emission line with intensity mapping, i.e. measuring unresolved emission at low-resolution but large cosmological volume, are poised to become one of the leading methods for tests of cosmological models on the largest scales, utilising instruemts such as MeerKAT [@Santos2016], HIRAX [@Newburgh2016], and the Square Kilometre Array [@Dewdney2009]. Foreground contamination is a common challenge amongst all such IM experiments, however, with Galactic and extragalactic foreground emission being several orders of magnitude brighter than the HI emission, necessitating the use of statistical or blind foreground separation techniques, which themselves must be calibrated on simulations.

A number of 21cm simulation packages exist, but most are not designed for the rapid exploration of post-EoR foregrounds and noise systematics needed when testing new analysis and calibration pipelines. `nbodykit` [@Hand2018] offers the ability to efficiently generate large scale cosmological structure, but `FastBox` aims to build on this by implementing foreground models, instrumental noise, and IM-specific observation effects. It is lightweight, and integrates with `pyccl` [@Chisari2019] for accurate nonlinear power spectra, allowing for a fully enclosed suite capable of benchmarking pipelines.

# State of the Field

`FastBox` intends to serve as a complementary package to existing cosmology-focused packages. `PowerBox` [@Murray2018] simulates two-point distributions (power spectra) in arbitrary numbers of dimensions, and is primarily intended to be a generator of mock galaxy distributions. `Tools21cm` [@Giri2020] aims to instead analyse simulated 21cm signals, primarily at the EoR and Cosmic Dawn (CD). For example, using previously and externally produced simulations, mock radio observations can be produced, as well as 21cm lightcones, and 1D, 2D, and cross power spectra. Lastly, `21cmFAST` [@REF] is a simulator focused on early-Universe fields, namely the EoR and CD. `FastBox` fills a niche in that it combines post-EoR signal simulation, foreground modelling, noise, and foreground filtering into a single lightweight package, providing an end-to-end test-bench specifically for IM analysis development.

# Software Design

`FastBox` is built around the `CosmoBox` class (`fastbox.box`), which contains the cosmological parameters, co-moving volume, and grid resolution of a simulation. This handles operations such as density field generation, and brightness temperature scaling. `pyccl` [@Chisari2019] handles cosmological computations, including correlation functions.

The package is organised into the following submodules:

- `fastbox.box` – core simulation box; density fields, redshift-space transforms, binned power spectra
- `fastbox.tracers` – HI tracer biasing, mean brightness temperature, and mock signal generation
- `fastbox.foregrounds` – Galactic synchrotron and extragalactic point source foreground models
- `fastbox.noise` – radiometer noise model for multi-dish arrays
- `fastbox.filters` – foreground separation (PCA, ICA, NMF) and transfer function estimation
- `fastbox.forecast` – Fisher matrix forecasts for cosmological parameters
- `fastbox.voids` – void detection and catalogue generation
- `fastbox.beams` – FFT and direct beam convolutions
- `fastbox.inpaint` – Gaussian process inpainting of flagged or missing data

All Fourier operations use `numpy.fft`, with `nbodykit` [@Hand2018] providing power spectrum multipoles and two-point correlation functions via its FFT routines.

# Usage Examples

The following illustrates a complete end-to-end simulation and analysis pipeline, beginning with signal generation and ending with power spectrum estimation and correlation function measurement.

## Generating a simulation box

A 128$^3$-cell box spanning $(2\,\mathrm{Gpc})^3$ at redshift $z = 0.8$ with an HI tracer, including a log-normal density transform and linear plus nonlinear redshift-space distortions:

```python
import numpy.fft as fft
import fastbox
from fastbox.box import CosmoBox, default_cosmo

box = CosmoBox(cosmo=default_cosmo, box_scale=(2e3, 2e3, 2e3),
               nsamp=128, redshift=0.8, realise_now=False)
box.realise_density()

# Apply HI bias and log-normal transform
tracer = fastbox.tracers.HITracer(box)
delta_hi = box.delta_x * tracer.bias_HI()
delta_ln = box.lognormal(delta_hi)

# Compute radial velocity field and transform to redshift space
# (sigma_nl = 120 km/s accounts for nonlinear finger-of-god smearing)
vel_k = box.realise_velocity(delta_x=box.delta_x, inplace=True)
vel_z = fft.ifftn(vel_k[2]).real
delta_s = box.redshift_space_density(delta_x=delta_ln.real,
                                     velocity_z=vel_z, sigma_nl=120.)

# Scale by mean brightness temperature to obtain signal cube in mK
signal_cube = tracer.signal_amplitude() * (1. + delta_s)
```

## Adding foregrounds and instrument noise

Galactic synchrotron emission and extragalactic point sources are modelled as spatially correlated maps with power-law spectral energy distributions, following the parameterisation of @Santos2005:

```python
from fastbox.foregrounds import ForegroundModel

fg = ForegroundModel(box)

# Galactic synchrotron (~133 K monopole at 130 MHz)
fg_synch_map  = fg.realise_foreground_amp(amp=700., beta=-2.4,
                                          monopole=133e3)
alpha_synch   = fg.realise_spectral_index(mean_spec_idx=-2.8,
                                          std_spec_idx=0.00002,
                                          smoothing_scale=0.1)
fg_synch_cube = fg.construct_cube(fg_synch_map, alpha_synch, freq_ref=130.)

# Extragalactic point sources (~26.7 K monopole at 130 MHz)
fg_ps_map  = fg.realise_foreground_amp(amp=57., beta=-1.1,
                                       monopole=26.7e3, smoothing_scale=0.1)
alpha_ps   = fg.realise_spectral_index(mean_spec_idx=-2.07,
                                       std_spec_idx=0.00002,
                                       smoothing_scale=0.1)
fg_ps_cube = fg.construct_cube(fg_ps_map, alpha_ps, freq_ref=130.)

# Radiometer noise for a MeerKAT-like 64-dish deep integration
noise_model = fastbox.noise.NoiseModel(box)
noise_cube  = noise_model.realise_radiometer_noise(Tinst=18., tp=0.25,
                                                   fov=1., Ndish=64)

data_cube = signal_cube + fg_synch_cube + fg_ps_cube + noise_cube
```

## Foreground removal and transfer function estimation

PCA, ICA, and NMF foreground filters are available through a unified interface. A bias-correction transfer function can be estimated via mock signal injection following @Cunnington2023:

```python
import functools

# Remove N_fg=3 foreground modes with PCA
cleaned_pca, U_fg, amp_fg = fastbox.filters.pca_filter(data_cube, nmodes=3,
                                                        return_filter=True)
# Alternative filters
cleaned_ica, _ = fastbox.filters.ica_filter(data_cube, nmodes=3,
                                             return_filter=True)
cleaned_nmf, _ = fastbox.filters.nmf_filter(data_cube, nmodes=3,
                                             return_filter=True)

# Estimate PCA transfer function using 100 mock signal injections
mock_fn = functools.partial(fastbox.tracers.generate_hi_mock,
                            cosmo=default_cosmo,
                            box_scale=(2e3, 2e3, 2e3),
                            nsamp=128, redshift=0.8)
T_s, T_m = fastbox.filters.pca_transfer_function(data_cube, cleaned_pca,
                                                  mock_fn, box,
                                                  nmodes=3, nmocks=100,
                                                  nbins=50)
```

## Power spectrum and correlation function estimation

```python
import numpy as np
from nbodykit.lab import ArrayMesh
from nbodykit.algorithms.fftcorr import FFTCorr

# Binned spherically-averaged power spectrum (transfer-function corrected)
k, pk, stddev = box.binned_power_spectrum(delta_x=cleaned_pca, nbins=50)
pk_corrected  = pk / T_m

# Theoretical prediction for comparison
th_k, th_pk = box.theoretical_power_spectrum()
amp_fac = (tracer.signal_amplitude() * tracer.bias_HI())**2.

# Two-point correlation function via nbodykit
boxsize  = (box.Lx, box.Ly, box.Lz)
mesh     = ArrayMesh(signal_cube, BoxSize=boxsize)
corrfn   = FFTCorr(first=mesh, mode='1d', BoxSize=boxsize,
                   los=[0, 0, 1], dr=2., rmin=20., rmax=200.)
corr, _  = corrfn.run()
```

# Research Impact

FastBox has been used in [@Murphy2026] to validate the statistical separation of 21cm signal and foregrounds in an IM-like experiment using Gibbs sampling and Gaussian Constrained Realisations.

# AI Usage Disclosure

Generative AI was used to create an initial outline of this manuscript in keeping with JOSS requirements, which the authors thereafter verified and further iterated on. The transfer function correction code was originally written for [@Murphy2026]. Generative AI was used to adapt this code in a manner consistent with all other `FastBox` functions, for example the inclusion of the docstring.

# Acknowledgements

[PLACEHOLDER: Acknowledge funding sources (grant numbers), computing resources, and contributors who are not listed as authors.]

# References
