#!/usr/bin/env python

import numpy as np
import pylab as plt
from fastbox.box import CosmoBox, default_cosmo
from nbodykit.algorithms.fftpower import FFTPower
from nbodykit.lab import ArrayMesh
from numpy import fft
import sys

# Use linear matter power for log-normal field
#default_cosmo['matter_power_spectrum'] = 'linear'

# Gaussian box
np.random.seed(11)
box = CosmoBox(cosmo=default_cosmo, box_scale=(1e2, 1e2, 1e2), nsamp=64, realise_now=False)
box.realise_density()

re_k, re_pk, re_stddev = box.binned_power_spectrum()
th_k, th_pk = box.theoretical_power_spectrum()


plt.matshow(box.delta_x[0].real, vmin=-1., vmax=5., cmap='cividis')
plt.title("Density field (real)")
plt.colorbar()

plt.matshow(box.delta_x[0].imag, vmin=-1., vmax=5., cmap='cividis')
plt.title("Density field (imag)")
plt.colorbar()
plt.show()

sys.exit(0)


# Trim negative values
y = np.log10(1.+box.delta_x[0])
y[np.isnan(y)] = 0.
plt.matshow(y, vmin=-2., vmax=3., cmap='cividis')
plt.title("Density field")
plt.colorbar()

# Log-normal field and power spectrum
delta_log = box.lognormal(box.delta_x)
logn_k, logn_pk, logn_stddev = box.binned_power_spectrum(delta_x=delta_log)


# NBodyKit version of power spectrum
mesh = ArrayMesh(delta_log, BoxSize=(box.Lx, box.Ly, box.Lz))
fft_power = FFTPower(mesh, mode='1d', Nmesh=None, 
                     BoxSize=(box.Lx, box.Ly, box.Lz), 
                     los=[0, 0, 1], Nmu=5, dk=None, kmin=0.0, kmax=None, poles=[])
lnpk_data, _ = fft_power.run()


plt.matshow(np.log10(1.+delta_log[0]), vmin=-2., vmax=3., cmap='cividis')
plt.title("Log-normal field")
plt.colorbar()

#plt.plot(th_k, th_pk, 'k-')
#plt.plot(re_k, re_pk, 'r.')
#plt.plot(logn_k, logn_pk, 'bx')
#plt.plot(lnpk_data['k'], lnpk_data['power'], 'g+')
#plt.xscale('log')
#plt.yscale('log')
plt.show()


sys.exit(0)

# Gaussian box with beam smoothing and foreground cut
transfer_fn = lambda k_perp, k_par: \
    (1. - np.exp(-0.5 * (k_par/0.001)**2.)) \
    * np.exp(-0.5 * (k_perp/0.1)**2.)
delta_smoothed = box.apply_transfer_fn(box.delta_k, transfer_fn=transfer_fn)


plt.matshow(delta_smoothed[:,:,0].real, vmin=-1., vmax=2., cmap='cividis')
plt.title("Density field (smoothed, x,y)")
plt.colorbar()
#plt.show()

plt.matshow(delta_smoothed[:,0,:].real, vmin=-1., vmax=2., cmap='cividis')
plt.title("Density field (smoothed, x,z)")
plt.colorbar()
#plt.show()

# Log-normal box
smre_k, smre_pk, smre_stddev \
    = box.binned_power_spectrum(delta_k=fft.fftn(delta_smoothed))

# Plot some stuff
fig = plt.figure()
plt.plot(th_k, th_pk, 'b-', label="Theoretical P(k)")
#plt.errorbar(re_k, re_pk, yerr=re_stddev, fmt=".", color='r')
plt.errorbar(re_k, re_pk, yerr=re_stddev, marker='.', color='r', 
             ls='none', label="P(k) from density field")
plt.errorbar(smre_k, smre_pk, yerr=smre_stddev, marker='x', color='g', 
             ls='none', label="P(k) from smoothed field")
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower left', frameon=False)
plt.ylim((1e0, 1e5))
plt.show()


sys.exit(0)


# Log-normal box
delta_ln = box.lognormal(box.delta_x)
lnre_k, lnre_pk, lnre_stddev \
    = box.binned_power_spectrum(delta_k=fft.fftn(delta_ln))

plt.matshow(delta_ln[0], vmin=-1., vmax=2., cmap='cividis')
plt.title("Log-normal density")
plt.colorbar()


# Tests
box.test_sampling_error()
box.test_parseval()


# Plot some stuff
fig = plt.figure()
plt.plot(th_k, th_pk, 'b-', label="Theoretical P(k)")
#plt.errorbar(re_k, re_pk, yerr=re_stddev, fmt=".", color='r')
plt.plot(re_k, re_pk, 'r.', label="P(k) from density field")
plt.plot(lnre_k, lnre_pk, 'gx', label="P(k) from log-normal")
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower left', frameon=False)

"""
plt.subplot(212)

def dx(R):
    dk = np.reshape(box.delta_k, np.shape(box.k))
    dk = dk * box.window1(box.k, R/box.cosmo['h'])
    dk = np.nan_to_num(dk)
    dx = fft.ifftn(dk)
    return dx

dx2 = dx(8.0)
dx3 = dx(100.0)

plt.hist(dx2.flatten(), bins=100, alpha=0.2)
plt.hist(dx3.flatten(), bins=100, alpha=0.2)
"""
plt.show()

