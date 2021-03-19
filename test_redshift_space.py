#!/usr/bin/env python

import numpy as np
import pylab as plt
from fastbox.box import CosmoBox, default_cosmo
from numpy import fft

# Gaussian box
np.random.seed(10)
box = CosmoBox(cosmo=default_cosmo, box_scale=(1e2, 1e2, 1e2), nsamp=128, realise_now=False)
box.realise_density()
box.realise_velocity()

# Plot real-space density field 
plt.matshow(box.delta_x[0], vmin=-1., vmax=20., cmap='cividis')
plt.title("Real-space")
plt.colorbar()

# Get redshift-space density field
vel_z = fft.ifftn(box.velocity_k[2]).real

delta_s = box.redshift_space_density(delta_x=box.delta_x, velocity_z=vel_z, 
                                     method='linear')

plt.matshow(delta_s[0], vmin=-1., vmax=20., cmap='cividis')
plt.title("Redshift-space")
plt.colorbar()

plt.show()
