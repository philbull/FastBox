#!/usr/bin/env python

import numpy as np
import pylab as plt
from fastbox.box import CosmoBox, default_cosmo
from fastbox.halos import HaloDistribution

# Gaussian box
np.random.seed(10)
box = CosmoBox(cosmo=default_cosmo, box_scale=(1e3, 1e3, 1e3), 
               nsamp=128, realise_now=False)
box.realise_density()

# Gaussian box with beam smoothing and foreground cut
transfer_fn = lambda k_perp, k_par: \
    (1. - np.exp(-0.5 * (k_par/0.00001)**2.)) \
    * np.exp(-0.5 * (k_perp/0.05)**2.)
delta_smoothed = box.apply_transfer_fn(box.delta_k, transfer_fn=transfer_fn)

plt.matshow(box.delta_x[0,:,:], vmin=-1., vmax=10., cmap='cividis')
plt.title("Density field (smoothed)")
plt.colorbar()


# Create halo distribution
halos = HaloDistribution(box, mass_range=(1e12, 1e15), mass_bins=10)
Nhalos = halos.generate_halos(box.delta_x, nbar=1e-4)

#re_k, re_pk, re_stddev = box.binned_power_spectrum()
#th_k, th_pk = box.theoretical_power_spectrum()

# Plot thresholded halos
Nh = Nhalos.copy()
Nh[np.where(Nh > 0.)] = 1.
Nh[np.where(Nh == 0.)] = 0.

print("Halos:", np.min(Nh), np.max(Nh))

plt.matshow(Nh[0,:,:], cmap='cividis')
plt.title("Halo count")
plt.colorbar()

plt.show()
