"""Generate figures for the FastBox JOSS paper."""
import numpy as np
import numpy.fft as fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyccl as ccl

import fastbox
from fastbox.box import CosmoBox, default_cosmo
from fastbox.foregrounds import ForegroundModel
from nbodykit.lab import ArrayMesh
from nbodykit.algorithms.fftcorr import FFTCorr

np.random.seed(41)

# ---------------------------------------------------------------------------
# (1) Generate simulation box
# ---------------------------------------------------------------------------
box = CosmoBox(cosmo=default_cosmo, box_scale=(2e3, 2e3, 2e3), nsamp=128,
               redshift=0.8, realise_now=False)
box.realise_density()

tracer = fastbox.tracers.HITracer(box)
delta_hi = box.delta_x * tracer.bias_HI()
delta_ln = box.lognormal(delta_hi)

vel_k = box.realise_velocity(delta_x=box.delta_x, inplace=True)
vel_z = fft.ifftn(vel_k[2]).real
delta_s = box.redshift_space_density(delta_x=delta_ln.real, velocity_z=vel_z,
                                     sigma_nl=120., method='linear')
signal_cube = tracer.signal_amplitude() * (1. + delta_s)

print("Step 1: simulation box done")

# ---------------------------------------------------------------------------
# (2) Foregrounds + noise
# ---------------------------------------------------------------------------
fg = ForegroundModel(box)

fg_synch_map  = fg.realise_foreground_amp(amp=700., beta=-2.4, monopole=133e3)
alpha_synch   = fg.realise_spectral_index(mean_spec_idx=-2.8, std_spec_idx=0.00002,
                                          smoothing_scale=0.1)
fg_synch_cube = fg.construct_cube(fg_synch_map, alpha_synch, freq_ref=130.)

fg_ps_map  = fg.realise_foreground_amp(amp=57., beta=-1.1, monopole=26.7e3,
                                       smoothing_scale=0.1)
alpha_ps   = fg.realise_spectral_index(mean_spec_idx=-2.07, std_spec_idx=0.00002,
                                       smoothing_scale=0.1)
fg_ps_cube = fg.construct_cube(fg_ps_map, alpha_ps, freq_ref=130.)

noise_model = fastbox.noise.NoiseModel(box)
noise_cube  = noise_model.realise_radiometer_noise(Tinst=18., tp=0.25,
                                                   fov=1., Ndish=64)
data_cube = signal_cube + fg_synch_cube + fg_ps_cube + noise_cube

print("Step 2: foregrounds + noise done")

# ---------------------------------------------------------------------------
# (3) Foreground cleaning
# ---------------------------------------------------------------------------
cleaned_pca, _, _ = fastbox.filters.pca_filter(data_cube, nmodes=3,
                                                return_filter=True)
cleaned_ica, _    = fastbox.filters.ica_filter(data_cube, nmodes=3,
                                                return_filter=True)
print("Step 3: foreground cleaning done")

# ---------------------------------------------------------------------------
# (3b) PCA signal-loss transfer function, via mock signal injection
# ---------------------------------------------------------------------------
# Each mock is a fresh 128^3 realisation plus a re-clean (~9 s), so nmocks=20
# takes roughly 3 min. The nmocks=100 default converges tighter but costs ~15.
mock_fn = lambda: fastbox.tracers.generate_hi_mock(default_cosmo,
                                                   box_scale=(2e3, 2e3, 2e3),
                                                   nsamp=128, redshift=0.8)
T_s, T_m = fastbox.filters.pca_transfer_function(data_cube, cleaned_pca, mock_fn,
                                                 box, nmodes=3, nmocks=20,
                                                 nbins=50)
print("Step 3b: transfer function done")

# ---------------------------------------------------------------------------
# Figure 1: slice of delta_ln and T_b (signal_cube)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

im0 = axes[0].matshow(delta_ln[10, :, :].T, vmin=-1., vmax=5.,
                      cmap='cividis', aspect='auto', origin='lower')
axes[0].set_xlabel(r"$x$ [cell]", fontsize=13)
axes[0].set_ylabel(r"$y$ [cell]", fontsize=13)
axes[0].set_title(r"Log-normal overdensity $\delta_{\rm ln}$", fontsize=13, pad=8)
axes[0].xaxis.set_ticks_position('bottom')
cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.set_label(r"$\delta_{\rm ln}$", fontsize=13)

im1 = axes[1].matshow(signal_cube[10, :, :].T, vmin=0., vmax=2.,
                      cmap='magma', aspect='auto', origin='lower')
axes[1].set_xlabel(r"$x$ [cell]", fontsize=13)
axes[1].set_ylabel(r"$y$ [cell]", fontsize=13)
axes[1].set_title(r"HI brightness temperature $T_b$", fontsize=13, pad=8)
axes[1].xaxis.set_ticks_position('bottom')
cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.set_label(r"$T_b(\mathbf{x})$ [mK]", fontsize=13)

fig.tight_layout()
fig.savefig("figures/field_slice.pdf", bbox_inches='tight', dpi=150)
fig.savefig("figures/field_slice.png", bbox_inches='tight', dpi=150)
print("Figure 1 saved: figures/field_slice.pdf")

# ---------------------------------------------------------------------------
# Figure 2: power spectra
# ---------------------------------------------------------------------------
sig_k,  sig_pk,  sig_err  = box.binned_power_spectrum(delta_x=signal_cube,   nbins=50)
pca_k,  pca_pk,  pca_err  = box.binned_power_spectrum(delta_x=cleaned_pca,   nbins=50)
ica_k,  ica_pk,  ica_err  = box.binned_power_spectrum(delta_x=cleaned_ica,   nbins=50)
th_k,   th_pk             = box.theoretical_power_spectrum()

amp_fac = (tracer.signal_amplitude() * tracer.bias_HI())**2.

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(th_k, th_pk * amp_fac, 'k-', lw=1.5, label="Theoretical $P(k)$", zorder=5)
ax.errorbar(sig_k, sig_pk, yerr=sig_err, color='steelblue', fmt='.',
            ms=5, capsize=2, label="True HI $P(k)$", zorder=4)
ax.errorbar(pca_k, pca_pk, yerr=pca_err, color='tomato', fmt='x',
            ms=5, capsize=2, label="PCA-cleaned $P(k)$", zorder=3)
ax.errorbar(ica_k, ica_pk, yerr=ica_err, color='goldenrod', fmt='s',
            ms=4, capsize=2, label="ICA-cleaned $P(k)$", zorder=2, alpha=0.8)

# T(k) is only meaningful where it is positive and finite: the top bins are
# empty because box.kmax overshoots the grid Nyquist, and the lowest-k bins
# hold few modes and can scatter negative.
tf_ok = np.isfinite(T_m) & (T_m > 0.)
ax.errorbar(pca_k[tf_ok], pca_pk[tf_ok] / T_m[tf_ok],
            yerr=pca_err[tf_ok] / T_m[tf_ok], color='mediumorchid', fmt='d',
            ms=4, capsize=2, label="PCA-cleaned, $T(k)$-corrected", zorder=6)
print("    T(k) usable in %d / %d bins" % (tf_ok.sum(), T_m.size))

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-3, 1.)
ax.set_ylim(1e1, 1e5)
ax.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]", fontsize=14)
ax.set_ylabel(r"$P(k)$ [$\mathrm{mK}^2\,h^{-3}\,\mathrm{Mpc}^3$]", fontsize=14)
ax.legend(frameon=False, fontsize=12)
ax.tick_params(which='both', direction='in', top=True, right=True)

fig.tight_layout()
fig.savefig("figures/power_spectrum.pdf", bbox_inches='tight', dpi=150)
fig.savefig("figures/power_spectrum.png", bbox_inches='tight', dpi=150)
print("Figure 2 saved: figures/power_spectrum.pdf")
