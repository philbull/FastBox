#!/usr/bin/env python
"""
Calculate the correlation function monopole over the data.
"""
import numpy as np
import pylab as plt
from fastbox.box import CosmoBox, default_cosmo
from numpy import fft
import pyccl as ccl
from nbodykit.lab import ArrayMesh
#from nbodykit.algorithms.paircount_tpcf.tpcf import SimulationBox2PCF
from nbodykit.algorithms.fftcorr import FFTCorr
from nbodykit.algorithms.fftpower import FFTPower
import time

# Use linear matter power for log-normal field
default_cosmo['matter_power_spectrum'] = 'linear'

# Gaussian box
np.random.seed(10)
box = CosmoBox(cosmo=default_cosmo, box_scale=(1e3, 1e3, 1e3), nsamp=128, 
               realise_now=False)
box.realise_density()
box.realise_velocity()


# Log-normal field and power spectrum
delta_log = box.lognormal(box.delta_x, transform_type='nbodykit')
#logn_k, logn_pk, logn_stddev = box.binned_power_spectrum(delta_x=delta_log)


# Convert to nbodykit mesh
mesh = ArrayMesh(box.delta_x, BoxSize=box.Lx)
mesh2 = ArrayMesh(delta_log, BoxSize=box.Lx)


# Putting BoxSize in Mpc units (instead of Mpc/h) produces output in Mpc units
corrfn = FFTCorr(first=mesh, mode='1d', BoxSize=(box.Lx, box.Ly, box.Lz), 
                 second=None, los=[0, 0, 1], Nmu=5, dr=2., rmin=10.0, rmax=200., 
                 poles=[])
corr, _ = corrfn.run()

# Log-normal
corrfn2 = FFTCorr(first=mesh2, mode='1d', BoxSize=(box.Lx, box.Ly, box.Lz), 
                 second=None, los=[0, 0, 1], Nmu=5, dr=2., rmin=10.0, rmax=200., 
                 poles=[])
corr2, _ = corrfn2.run()


# Output correlation function
print(corr)

r = corr['r']
rr = np.linspace(2., 200., 300)
h = box.cosmo['h']

plt.subplot(211)
plt.plot(rr, rr**2. * ccl.correlation_multipole(box.cosmo, a=1., l=0, s=rr, beta=0.), 'k-')
plt.plot((r), (r)**2. * corr['corr'], 'r.')
plt.plot((r), (r)**2. * corr2['corr'], 'bx')
plt.xlabel("r", fontsize=16)
plt.ylabel(r"$r^2 \xi(r)$", fontsize=16)


plt.subplot(212)
k = np.logspace(-3., 0., 200)
plt.plot(k, ccl.linear_matter_power(box.cosmo, a=1., k=k), 'k-')
plt.plot(k, ccl.nonlin_matter_power(box.cosmo, a=1., k=k), 'r--')
plt.xlabel("k", fontsize=16)
plt.ylabel("P(k)", fontsize=16)
plt.xscale('log')
plt.yscale('log')

plt.show()
