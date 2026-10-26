---
title: 'FastBox: A lightweight Python package for rapid cosmological 21cm intensity mapping simulations'
tags:
  - Python
  - cosmology
  - 21cm intensity mapping
  - signal simulations
  - foreground removal
authors:
  - name: Philip Bull
    orcid: 0000-0001-5668-3101
    affiliation: 1   # TODO: confirm current affiliation
  - name: Bruno B. Bizarria
    orcid: 0000-0001-7794-6599
    affiliation: 2   # TODO: confirm current affiliation
  - name: Melis O. Irfan
    orcid: 0000-0003-2021-7357
    affiliation: 3   # TODO: confirm current affiliation
  - name: Geoff Murphy
    orcid: 0000-0002-8186-3064
    affiliation: 4   # TODO: confirm current affiliation
affiliations:
  - name: "Jodrell Bank Centre for Astrophysics, Department of Physics & Astronomy, The University of Manchester, Manchester M13 9PL, UK"
    index: 1
  - name: "Instituto Nacional de Pesquisas Espacias, São José dos Campos, Brazil"
    index: 2
  - name: "Institute of Astrophysics, University of Cambridge, Madingley Road, CB3 0HA, UK"
    index: 3
  - name: "Department of Physics & Astronomy, University of the Western Cape, Robert Sobukwe Road, Cape Town 7535, South Africa"
    index: 4
date: 22 October 2026
bibliography: paper.bib
---

# Summary

`FastBox` is a Python package for generating fast, physically realistic simulations of cosmological signals in three-dimensional co-moving boxes, with the primary focus of its application being 21cm intensity mapping (IM) experiments. It provides a framework for producing cosmology-dependent Gaussian and log-normal density fields, as well as modelling the effects of redshift-space distortions and linear biasing, among others. Models of instrumental systematics are incorporated, including radiometer noise and beam convolutions. Diffuse and point source foreground models are included, along with a number of foreground filtering strategies such as PCA, ICA, and transfer function correction via mock signal injection. Lastly, estimators are provided for power spectra, the two-point correlation function, and the equilateral bispectrum. The same underlying density field can also be Poisson-sampled into a discrete galaxy catalogue, so that 21cm and galaxy tracers are simulated from common initial conditions and their cross-power spectrum can be measured. Simulation grids may be cubic or anisotropic, with independent side lengths and cell counts along each Cartesian axis. `FastBox` is designed as a lightweight but realistic test-bench for the development and validation of end-to-end cosmological analysis pipelines.

# Statement of Need

Intensity mapping is a way of making 3D maps of the redshifted 21cm emission line from neutral hydrogen (HI). It works by measuring the combined emission from many sources containing neutral hydrogen gas at low angular resolution, but over a large cosmological volume. 21cm IM surveys are a promising method for tests of cosmological models on the largest distance scales in the post-reionisation era, utilising instruments such as CHIME [@CHIME2025], HIRAX [@Newburgh2016], GBT [@2010arXiv1007.3709C], MeerKAT [@2026arXiv261012158M], Parkes [@2018MNRAS.476.3382A], and eventually the Square Kilometre Array [@2026aask.bookE.106W]. Foreground contamination is a common challenge for all such IM experiments however, with Galactic and extragalactic foreground emission being several orders of magnitude brighter than the 21cm emission. The foregrounds are further corrupted by frequency-dependent instrumental effects, such as the telescope beam. This necessitates the use of statistical or data-driven foreground removal techniques, which themselves need to be tested on simulations to mitigate issues such as signal loss (overfitting).

While the performance of 21cm surveys has often been predicted using techniques such as Fisher matrix forecasts [@2015ApJ...803...21B], these are simplistic. Approaches such as simulated mock analyses provide more realism, and can more accurately take into account the systematic effects that dominate the error budget for this type of experiment. These analyses are necessarily quite complex however, and require a way to simulate data with the correct cosmological correlations and foreground properties, add instrument-dependent systematic effects, and then apply a variety of data analysis methods such as foreground removal, signal loss correction, and power spectrum estimation in order to understand the likely real-world performance of any given experiment.

# State of the Field

A number of 21cm simulation packages exist (e.g. [2020MNRAS.499..304C], [@2022JCAP...05..002R], [@2026JCAP...01..019M], [@2026arXiv260701864C]), each making a variety of different approximations and assumptions about the 21cm field and instrumental effects. Depending on the application, these may or may not be suitable, e.g. for the rapid exploration of cosmological models, testing of gridding, map-making, beam convolution, and foreground removal effects, and different types of noise and residual systematics. Several or even all of these features may be needed when testing new analysis and calibration pipelines however. In addition, these codes often provide only the simulation tools, with basic data analysis steps left to external packages. This results in additional complexity for many applications, as multiple codes must be installed and interfaced to perform the semi-realistic analyses that are needed to validate different survey designs or analysis approaches.

# Software Design

`FastBox` has been designed to provide a 'one-stop' solution to the problem of making semi-realistic 21cm intensity mapping data and subsequently performing the main analysis steps needed by real-world surveys. The package provides multiple fast cosmological simulation options; a range of realistic foreground models; various options for adding instrumental noise and realistic beam convolution effects; and features such as correlated galaxy field generation, void identification, and power spectrum and bispectrum estimation. It has few dependencies, integrates with the widely-used `pyccl` cosmology code [@Chisari2019] for accurate cosmology-dependent nonlinear power spectra, and provides a fully self-contained suite for building and running mock data generation and analysis pipelines. While some of the chosen methods (e.g. log-normal density field simulation) are approximate, they are fast and are implemented directly within `FastBox`, rather than adding to the list of dependencies.

`FastBox` is built around the `CosmoBox` class (`fastbox.box`), which contains the cosmological parameters, co-moving volume, and grid resolution of a simulation. This handles operations such as density field generation, and brightness temperature scaling. The box need not be cubic: both the side lengths and the number of grid cells may be specified independently per Cartesian axis, so that a simulated volume can be matched to the anisotropic sky-frequency geometry of a real IM survey. `pyccl` [@Chisari2019] handles theoretical cosmological computations.

The package is organised into the following submodules:

- `fastbox.box` – core simulation box; density fields, redshift-space transforms, cubic and anisotropic grids
- `fastbox.power` – auto, cross and weighted power spectrum estimation, forward-modelled observational power spectra, the 1D two-point correlation function, and the equilateral bispectrum
- `fastbox.tracers` – HI tracer biasing, mean brightness temperature, mock signal generation, and Poisson-sampled galaxy catalogues
- `fastbox.foregrounds` – Galactic synchrotron and extragalactic point source foreground models
- `fastbox.noise` – radiometer noise model for multi-dish arrays
- `fastbox.filters` – foreground separation (PCA, ICA, NMF) and transfer function estimation
- `fastbox.forecast` – Fisher matrix forecasts for cosmological parameters
- `fastbox.voids` – void detection and catalogue generation
- `fastbox.halos` – halo catalogue generation
- `fastbox.beams` – beam models (Gaussian, KATBeam, Zernike) with FFT and direct convolutions
- `fastbox.inpaint` – Gaussian process inpainting of flagged or missing data
- `fastbox.meerklass` – MeerKLASS survey geometry, released-window loading, hit-map noise, and masked weighted PCA
- `fastbox.analysis`, `fastbox.plot`, `fastbox.utils` – analysis helpers, plotting, and shared utilities

All Fourier operations use `numpy.fft`. Power spectrum multipoles, the two-point correlation function, and the equilateral bispectrum are estimated natively in `fastbox.power`, using established estimators such as the Wiener-Khinchin theorem (for the correlation function) and the Scoccimarro estimator (for the bispectrum).

## Cross-correlation with galaxy surveys

Cross-correlating an intensity map with a galaxy survey is one of the main routes
to a 21cm detection, since foreground residuals do not correlate with galaxy
positions and so add scatter to the cross spectrum rather than bias. Simulating
this requires both tracers to be drawn from the same realisation of the density
field, which `fastbox.tracers` provides.

`GalaxyTracer` turns a density field into a discrete galaxy catalogue by Poisson
sampling, given a target number density and a field that already carries the
galaxy bias. Because the galaxies
are discrete objects rather than a continuous field, redshift-space distortions
can be applied to their positions directly, and the catalogue is then placed back
onto the simulation grid using a choice of mass-assignment scheme.

`fastbox.power` measures auto and cross spectra from the resulting maps,
optionally weighting cells to represent a survey's selection, and corrects for
the smoothing that gridding a catalogue introduces [@Jing2005]. Its theoretical
predictions are built on the same grid as the measurement rather than in the
continuum, so that the beam, the channel width and the gridding are applied to
the model instead of being removed from the data. Separate models are provided
for an intensity map (`predict_hi_power_spectrum`), a galaxy map
(`predict_galaxy_power_spectrum`), and their cross spectrum
(`predict_cross_spectrum`).

# Usage Examples

The following illustrates a complete end-to-end simulation and analysis pipeline, beginning with signal generation and ending with power spectrum estimation and correlation function measurement.

## Generating a simulation box

A 128$^3$-cell box spanning $(2\,\mathrm{Gpc})^3$ at redshift $z = 0.8$ with an HI tracer, including a log-normal density transform and linear plus nonlinear redshift-space distortions:

```python
import fastbox
from fastbox.box import CosmoBox, default_cosmo

box = CosmoBox(cosmo=default_cosmo, box_scale=(2e3, 2e3, 2e3),
               nsamp=128, redshift=0.8, realise_now=False)
box.realise_density()

# Apply the HI bias and linear redshift-space distortions in Fourier space
# (sigma_v = 120 km/s accounts for nonlinear finger-of-god smearing), then
# a log-normal transform so that the brightness temperature stays positive
tracer = fastbox.tracers.HITracer(box)
delta_hi = box.linear_rsd_density(box.delta_x, bias=tracer.bias_HI(),
                                  sigma_v=120.)
delta_ln = box.lognormal(delta_hi)

# Scale by mean brightness temperature to obtain signal cube in mK
signal_cube = tracer.signal_amplitude() * (1. + delta_ln)
```

![A single frequency slice ($z = 0.8$) of the log-normal overdensity field $\delta_{\rm ln}$ (left) and the corresponding HI brightness temperature $T_b$ (right), generated on a $128^3$ grid spanning $(2\,\mathrm{Gpc})^3$.](figures/field_slice.pdf){#fig:field_slice width=100%}

## Adding foregrounds and instrument noise

Galactic synchrotron emission and extragalactic point sources are modelled as spatially-correlated maps with power-law spectral energy distributions, following the parameterisation of @Santos2005:

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

PCA, ICA, and NMF foreground filters are available through a unified interface. A bias-correction transfer function can be estimated via mock signal injection, implementing the method introduced by @Cunnington2023:

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
from fastbox.power import Power

# Binned spherically-averaged power spectrum (transfer-function corrected)
k, pk, stddev = box.binned_power_spectrum(delta_x=cleaned_pca, nbins=50)
pk_corrected  = pk / T_m

# Forward-model the prediction onto the same grid, including RSD
power = Power(box)
th_k = np.logspace(-4, 1, 2000)
_, th_pk = power.predict_matter_power_spectrum(th_k)
k_im, pk_im, _ = power.predict_hi_power_spectrum(th_k, th_pk,
                                                 bias=tracer.bias_HI(),
                                                 Tb=tracer.signal_amplitude(),
                                                 sigdeg=0., sigma_nl=120.)

# Higher-order and configuration-space statistics
# 1D two-point correlation function (Wiener-Khinchin, no external dependency)
r, xi = power.estimate_correlation_function(signal_cube)

# Equilateral bispectrum B(k, k, k) via the Scoccimarro estimator
k_b, b_eq, ntri = power.estimate_equilateral_bispectrum(signal_cube, n_bins=8)
```

![Spherically-averaged power spectra from an end-to-end simulation at $z = 0.8$. The theoretical prediction (black) is compared against the true HI signal (blue), and the signal recovered after PCA (red) and ICA (yellow) foreground removal with $N_{\rm fg} = 3$ modes subtracted. The prediction is the linear forward model `predict_hi_power_spectrum`, so above $k \approx 0.05\,\mathrm{Mpc}^{-1}$ it falls below the true signal, to which the log-normal transform adds small-scale power. Large-scale power loss from foreground cleaning is visible at low $k$. Dividing the PCA result by the transfer function $T(k)$, estimated here from 20 mock signal injections, restores that lost power (purple). At high $k$ the corrected points lie above the true signal: $T(k)$ compensates for signal loss but not for the residual noise, which dominates the cleaned map on those scales.](figures/power_spectrum.pdf){#fig:power_spectrum width=100%}

## Galaxy catalogue and HI-galaxy cross spectrum

```python
from fastbox.tracers import GalaxyTracer

# Galaxy overdensity from the same density field, with its own bias and the
# same redshift-space distortions as the HI (GalaxyTracer expects the bias to
# be in the field already)
b_gal = 1.4
delta_g = box.lognormal(box.linear_rsd_density(box.delta_x, bias=b_gal,
                                               sigma_v=120.))

# Poisson-sample a galaxy catalogue and assign it to the grid as an
# overdensity, using cloud-in-cell weighting
gal = GalaxyTracer(box, vol_density=1e-3)
delta_gal = gal.generate_mesh(delta_g, method='CIC', overdensity=True)

# HI x galaxy cross spectrum (estimate_weighted_power_spectrum takes
# per-cell weights instead)
k, pk_cross, sigma_cross = power.estimate_power_spectrum(cleaned_pca,
                                                         delta_gal)

# Forward-model the prediction onto the same grid, with the CIC window applied
# to the model rather than deconvolved from the data. These maps have no beam;
# for beam-convolved maps, set sigdeg to the beam width in degrees
k_cc, pk_cc, _ = power.predict_cross_spectrum(th_k, th_pk,
                                              bias_HI=tracer.bias_HI(),
                                              bias_gal=b_gal,
                                              Tb=tracer.signal_amplitude(),
                                              sigdeg=0., MAS='CIC',
                                              sigma_nl=120.)
```

The forward models are linear in the density field. On large scales the measured
cross spectrum therefore falls below the model due to PCA signal loss, which the
transfer function above corrects; on small scales it rises above the model,
because the log-normal transform adds power that a linear model does not include.

# Research Impact

`FastBox` simulated 21cm data cubes were used in [@Irfan&Bull2021], where kernel principal component analysis was tested as an alternative to standard principal component analysis (PCA) foreground removal techniques. KPCA was found to improve recovery at intermediate-to-large cosmological scales. [@Irfan2023] generated mock 21cm cubes to test PCA foreground removal in four cases of Galactic synchrotron spectral index models, including a high-resolution model they generated via a convolutional neural network (CNN). [@Irfan2024] used `FastBox` mock signal and foreground cubes as inputs for pipelines focused on the removal of correlated noise.

[@Engelbrecht2025] used mock `FastBox` signal cubes to validate their Radio Navigation Satellite System Radio Frequency Interference (RFI) model against MeerKAT data. [@Murphy2026] used mock signal and foreground datacubes to validate the statistical separation of 21cm signal and foregrounds in an IM-like experiment using Gibbs sampling and Gaussian Constrained Realisations.

# AI Usage Disclosure

Generative AI was used to create an initial outline of this manuscript in-keeping with JOSS requirements, which the authors thereafter verified and further iterated on. The transfer function correction implements the mock signal injection method of @Cunnington2023, and is not a new method. The implementation contributed here was written independently for [@Murphy2026]; generative AI was then used to adapt it to `FastBox`'s conventions, for example the inclusion of the docstring. Lastly, AI was used to perform additional tests on the code base, with any bug and error reports being verified for validity by the authors before implementation.

# Acknowledgements

B.B.B. acknowledges resources received from Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP) under process 2022/16749-0. This result is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant agreement No. 948764; PB). 

# References
