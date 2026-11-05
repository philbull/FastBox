#!/usr/bin/env python

import numpy as np
import pylab as plt
from fastbox.box import CosmoBox, default_cosmo
from numpy import fft
import fastbox.power as power

# Gaussian box
np.random.seed(10)
box = CosmoBox(cosmo=default_cosmo, box_scale=(1e2, 1e2, 1e2), nsamp=128, realise_now=False)
box.realise_density()
box.realise_velocity()

#Compare different methods of including RSD effect

# Get redshift-space density field
vel_z = fft.ifftn(box.velocity_k[2]).real

delta_s_numerical = box.interpolated_rsd_density(delta_x=box.delta_x, velocity_z=vel_z, 
                                     sigma_nl=200., method='linear')


delta_s_analytical = box.linear_rsd_density(delta_x=box.delta_x, sigma_v=200., fog_model='gaussian')

#COMPARE POWER SPECTRUM
power = power.Power(box)
k, pk_num, _ = power.estimate_power_spectrum(delta_s_numerical)
k, pk_an , _ = power.estimate_power_spectrum(delta_s_analytical)
km,pkm = power.predict_matter_power_spectrum(k,rsd=True,sigma_nl=200)

plt.plot(km,pkm,color='k',label='Theory')
plt.plot(k, pk_num,label='Numerical')
plt.plot(k, pk_an,label='Analytical')
plt.loglog()
plt.legend()
plt.show()


# Plot real-space density field 
plt.matshow(box.delta_x[0], vmin=-1., vmax=20., cmap='cividis')
plt.title("Real-space")
plt.colorbar()

plt.matshow(delta_s_analytical[0], vmin=-1., vmax=20., cmap='cividis')
plt.title("linear_rsd_density")
plt.colorbar()

plt.show()

plt.matshow(delta_s_numerical[0], vmin=-1., vmax=20., cmap='cividis')
plt.title("interpolated_rsd_density")
plt.colorbar()

plt.show()
