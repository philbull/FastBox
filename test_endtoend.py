#!/usr/bin/env python
"""
Test a simple end-to-end simulation, generating a log-normal field, transforming 
to redshift-space, adding foregrounds and noise, removing foregrounds, and 
estimating the correlation function.
"""
import numpy as np
import numpy.fft as fft
import pylab as plt
import pyccl as ccl
import fastbox
from fastbox.box import CosmoBox, default_cosmo
from fastbox.foregrounds import ForegroundModel
from nbodykit.lab import ArrayMesh
from nbodykit.algorithms.fftcorr import FFTCorr
from nbodykit.algorithms.fftpower import FFTPower
import time, sys

#-------------------------------------------------------------------------------
# Realise density field in redshift space
#-------------------------------------------------------------------------------
print("(1) Generating box...")
t0 = time.time()

# (1a) Generate Gaussian box
np.random.seed(10)
box = CosmoBox(cosmo=default_cosmo, box_scale=(4e3,4e3,4e3), nsamp=128, 
               redshift=0.8, realise_now=False)
box.realise_density()

# (1b) Rescale tracer by bias [FIXME: Check this is being done in the right order]
tracer = fastbox.tracers.HITracer(box)
delta_hi = box.delta_x * tracer.bias_HI()

# (1c) Transform to a log-normal field
delta_ln = box.lognormal(delta_hi)

# (1d) Calculate radial velocity field (uses Gaussian density field; FIXME)
vel_k = box.realise_velocity(delta_x=box.delta_x, inplace=True)
vel_z = fft.ifftn(vel_k[2]).real # inverse FFT to get real-space radial velocity

# (1e) Transform to redshift space
delta_s = box.redshift_space_density(delta_x=delta_ln.real, velocity_z=vel_z, 
                                     sigma_nl=120., method='linear')

# (1f) Scale by mean brightness temperature (in mK), and include mean
signal_cube = tracer.signal_amplitude() * (1. + delta_s)

print("\t(1) Generating box complete (%3.3f sec)" % (time.time()-t0))

#-------------------------------------------------------------------------------
# Add foregrounds
#-------------------------------------------------------------------------------
print("(2) Adding foregrounds...")
t0 = time.time()

# Create new foreground model
fg = ForegroundModel(box)
fg_map = fg.realise_foreground_amp(amp=57., beta=1.1, monopole=10., 
                                   smoothing_scale=4., redshift=box.redshift) # in mK

# Construct spectral index map
ang_x, ang_y = box.pixel_array(redshift=box.redshift)
alpha = fg.realise_spectral_index(mean_spec_idx=2.07, std_spec_idx=0.0002,
                                  smoothing_scale=15., redshift=box.redshift)

# Construct foreground datacube
fg_cube = fg.construct_cube(fg_map, alpha, freq_ref=130., redshift=box.redshift)

# Add to data cube
data_cube = signal_cube + fg_cube

print("\t(2) Adding foregrounds complete (%3.3f sec)" % (time.time()-t0))

#-------------------------------------------------------------------------------
# Add noise
#-------------------------------------------------------------------------------
print("(3) Adding noise...")
t0 = time.time()

# Generate homogeneous radiometer noise
noise_model = fastbox.noise.NoiseModel(box)
noise_cube = noise_model.realise_radiometer_noise(Tinst=18., tp=2., fov=1., 
                                                  Ndish=64) # FIXME: Long integration time!

# Add to data cube
data_cube += noise_cube

print("\t(3) Adding noise complete (%3.3f sec)" % (time.time()-t0))

#-------------------------------------------------------------------------------
# Beam convolution
#-------------------------------------------------------------------------------

# FIXME


#-------------------------------------------------------------------------------
# Foreground cleaning
#-------------------------------------------------------------------------------
print("(4) Cleaning foregrounds...")
t0 = time.time()

# Apply PCA cleaning
cleaned_cube4, U_fg, amp_fg = fastbox.filters.pca_filter(data_cube, 
                                                         nmodes=4, 
                                                         return_filter=True)

cleaned_cube8, U_fg, amp_fg = fastbox.filters.pca_filter(data_cube, 
                                                         nmodes=12, 
                                                         return_filter=True)
#cleaned_cube = data_cube # FIXME

print("\t(4) Cleaning foregrounds complete (%3.3f sec)" % (time.time()-t0))
#-------------------------------------------------------------------------------
# Calculate correlation function
#-------------------------------------------------------------------------------
print("(5) Calculating correlation function...")
t0 = time.time()

# Convert fields to nbodykit meshes
boxsize = (box.Lx, box.Ly, box.Lz)
mesh_true = ArrayMesh(signal_cube, BoxSize=boxsize)
mesh_proc4 = ArrayMesh(cleaned_cube4, BoxSize=boxsize)
mesh_proc8 = ArrayMesh(cleaned_cube8, BoxSize=boxsize)

# Correlation function of true signal
corrfn_true = FFTCorr(first=mesh_true, mode='1d', BoxSize=boxsize, 
                      los=[0, 0, 1], dr=2., rmin=20.0, rmax=200.)
corr_true, _ = corrfn_true.run()

# Perform high-pass filter on foreground-cleaned data
highpass = lambda kperp, kpar: 1. - np.exp(-0.5 * (np.abs(kpar) / 0.009)**3.)
cleaned_cube4_hp = box.apply_transfer_fn(fft.fftn(cleaned_cube4), transfer_fn=highpass)
mesh_proc4_hp = ArrayMesh(cleaned_cube4_hp.real, BoxSize=boxsize)


# Correlation function of processed (FG-subtracted) signal
corrfn_proc4 = FFTCorr(first=mesh_proc4, mode='1d', BoxSize=boxsize, 
                       los=[0, 0, 1], dr=2., rmin=20.0, rmax=200.)
corr_proc4, _ = corrfn_proc4.run()

# 4 modes subtracted, high-pass filtered
corrfn_proc4_hp = FFTCorr(first=mesh_proc4_hp, mode='1d', BoxSize=boxsize, 
                       los=[0, 0, 1], dr=2., rmin=20.0, rmax=200.)
corr_proc4_hp, _ = corrfn_proc4_hp.run()

# 8 modes subtracted
corrfn_proc8 = FFTCorr(first=mesh_proc8, mode='1d', BoxSize=boxsize, 
                       los=[0, 0, 1], dr=2., rmin=20.0, rmax=200.)
corr_proc8, _ = corrfn_proc8.run()

print("\t(5) Calculating correlation function complete (%3.3f sec)" % (time.time()-t0))

#-------------------------------------------------------------------------------
# Plotting
#-------------------------------------------------------------------------------


# Plot fields
#plt.matshow(signal_cube[0,:,:], vmin=-1., vmax=2.)
#plt.title("Signal")
#plt.colorbar()


#plt.matshow(cleaned_cube4[0,:,:], vmin=-1., vmax=2.)
#plt.title("FG-subtracted (4 modes)")
#plt.colorbar()
##plt.show()

#plt.matshow(cleaned_cube4_hp[0,:,:].real, vmin=-1., vmax=2.)
#plt.title("FG-subtracted (4 modes, high-pass)")
#plt.colorbar()

#plt.matshow(cleaned_cube8[0,:,:], vmin=-1., vmax=2.)
#plt.title("FG-subtracted (8 modes)")
#plt.colorbar()


# Power spectra
sig_k, sig_pk, sig_stddev = box.binned_power_spectrum(delta_x=signal_cube)
proc4_k, proc4_pk, proc4_stddev = box.binned_power_spectrum(delta_x=cleaned_cube4)
proc8_k, proc8_pk, proc8_stddev = box.binned_power_spectrum(delta_x=cleaned_cube8)
proc4hp_k, proc4hp_pk, proc4hp_stddev = box.binned_power_spectrum(delta_x=cleaned_cube4_hp)
th_k, th_pk = box.theoretical_power_spectrum()

amp_fac = (tracer.signal_amplitude() * tracer.bias_HI())**2.

plt.figure()
plt.subplot(111)
plt.plot(th_k, th_pk * amp_fac, 'k-')
plt.errorbar(sig_k, sig_pk, yerr=sig_stddev, color='r', marker='.')
plt.errorbar(proc4_k, proc4_pk, yerr=proc4_stddev, color='b', marker='x')
plt.errorbar(proc4hp_k, proc4hp_pk, yerr=proc4hp_stddev, color='y', marker='s')
plt.errorbar(proc8_k, proc8_pk, yerr=proc8_stddev, color='g', marker='+')

plt.xscale('log')
plt.yscale('log')
#plt.show()

#sys.exit(0)

# Plot correlation functions and vanilla theoretical prediction
plt.figure()
plt.subplot(111)
r = corr_true['r']
h = box.cosmo['h']

rr = np.linspace(2., 200., 300)
xi = ccl.correlation_multipole(box.cosmo, a=box.scale_factor, l=0, s=rr, beta=0.)

plt.subplot(111)
plt.plot(rr, rr**2. * xi * tracer.signal_amplitude()**2., 'k-')
plt.plot(r, r**2. * corr_true['corr'], 'r.', label="True field")
plt.plot(r, r**2. * corr_proc4['corr'], 'bx', label="4 modes")
plt.plot(r, r**2. * corr_proc4_hp['corr'], 'ys', label="4 modes (high-pass)")
plt.plot(r, r**2. * corr_proc8['corr'], 'g+', label="4 modes")
plt.xlabel("r", fontsize=16)
plt.ylabel(r"$r^2 \xi(r)$", fontsize=16)

plt.show()
