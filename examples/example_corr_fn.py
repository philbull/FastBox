#!/usr/bin/env python
"""
Calculate the correlation function monopole over the data.
"""

import numpy as np
import pylab as plt
from fastbox.box import CosmoBox, default_cosmo
from numpy import fft
import pyccl as ccl
import time
import fastbox.power as power

# Use linear matter power for log-normal field
default_cosmo['matter_power_spectrum'] = 'linear'

# Gaussian box
np.random.seed(10)
box = CosmoBox(cosmo=default_cosmo, box_scale=(1e3, 1e3, 1e3), nsamp=128, 
               realise_now=False)
box.realise_density()
box.realise_velocity()


# Log-normal field
delta_log = box.lognormal(box.delta_x)


power_obj = power.Power(box)
r, corr = power_obj.estimate_correlation_function(box.delta_x)
r, corr2 = power_obj.estimate_correlation_function(delta_log)

corr = corr[r<200]
corr2 = corr2[r<200]

r=r[r<200]


h = box.cosmo['h']

plt.plot(r, r**2. * ccl.correlation_multipole(box.cosmo, a=1., ell=0, r=r, beta=0.), 'k-',label='Theory')
plt.plot(r, r**2. * corr, 'r.', label='Gaussian')
plt.plot(r, r**2. * corr2, 'bx', label='Lognormal')
plt.xlabel("r", fontsize=16)
plt.ylabel(r"$r^2 \xi(r)$", fontsize=16)
plt.legend()
plt.show()
