#!/usr/bin/env python

import numpy as np
import pylab as plt
from fastbox.box import CosmoBox, default_cosmo
from fastbox.halos import HaloDistribution
import fastbox.power as power
from fastbox.tracers import GalaxyTracer


# Gaussian box
np.random.seed(10)
box = CosmoBox(cosmo=default_cosmo, box_scale=(2e3, 2e3, 2e3), 
               nsamp=64, realise_now=False)
box.realise_density()

delta_ln = box.lognormal(delta_x=box.delta_x)

# Create halo distribution
halos = HaloDistribution(box, mass_range=(1e12, 1e15), mass_bins=10)
Nhalos = halos.halo_count_field(box.delta_x, nbar=1e-3, bias=1.)
halo_cat = halos.realise_halo_catalogue(Nhalos, scatter=True, scatter_type='uniform')



power_obj = power.Power(box)
gal_tracer = GalaxyTracer(box=box, vol_density=None, bias=None)
mesh = gal_tracer._assign_ngp(halo_cat)

k, pspec1, _ = power_obj.estimate_power_spectrum(mesh)
k, pspec2, _ = power_obj.estimate_power_spectrum(box.delta_x)
k, pspec3, _ = power_obj.estimate_power_spectrum(delta_x1=box.delta_x,delta_x2=mesh)



# Plot power spectra
plt.subplot(111)
plt.plot(k, pspec1, 'r.', 
         ls='solid', label="hh")
plt.plot(k, pspec2, 'bx', 
         ls='solid', label=r"$\delta\delta$ lin")
plt.plot(k, pspec3, 'y+', 
         ls='solid', label=r"$h\delta$")
plt.yscale('log')
#plt.ylim((1e2, 1e5))
plt.legend(loc='upper right', frameon=False)

plt.show()


#plot density field with halos
plt.figure()
plt.subplot(111)
plt.matshow(box.delta_x[0,:,:], vmin=-1., vmax=5., cmap='cividis', fignum=False)
plt.title("Density field (smoothed)")
plt.colorbar()

idxs = np.where(np.logical_and(halo_cat[:,2] >= 0., halo_cat[:,2] < 2.*box.Lz/box.N))
_x, _y, _z = halo_cat[idxs].T
plt.plot(_x*box.N/box.Lx, _y*box.N/box.Ly, marker='.', color='r', ms=0.05, ls='none')

print("Min/max:", np.min(halo_cat), np.max(halo_cat))

plt.title("Halo count")

plt.show()
