#!/usr/bin/env python

import numpy as np
import pylab as plt
from fastbox.box import CosmoBox, default_cosmo
from fastbox.halos import HaloDistribution

from nbodykit.lab import ArrayMesh, ArrayCatalog
from nbodykit.algorithms.fftcorr import FFTCorr
from nbodykit.algorithms.fftpower import FFTPower

# Gaussian box
np.random.seed(10)
box = CosmoBox(cosmo=default_cosmo, box_scale=(2e3, 2e3, 2e3), 
               nsamp=64, realise_now=False)
box.realise_density()

# Gaussian box with beam smoothing and foreground cut
#transfer_fn = lambda k_perp, k_par: \
#    (1. - np.exp(-0.5 * (k_par/0.00001)**2.)) \
#    * np.exp(-0.5 * (k_perp/0.05)**2.)
#delta_smoothed = box.apply_transfer_fn(box.delta_k, transfer_fn=transfer_fn)

delta_ln = box.lognormal(delta_x=box.delta_x)

# Create halo distribution
halos = HaloDistribution(box, mass_range=(1e12, 1e15), mass_bins=10)
Nhalos = halos.halo_count_field(box.delta_x, nbar=1e-3, bias=1.)
#Nhalos2 = halos.halo_count_field(box.delta_x, nbar=1e-3, bias=1.)
halo_cat = halos.realise_halo_catalogue(Nhalos, scatter=True, scatter_type='uniform')
#halo_cat2 = halos.realise_halo_catalogue(Nhalos, scatter=True, scatter_type='uniform')

# Project catalogue onto mesh
array_cat = ArrayCatalog({'Position': halo_cat})
#array_cat2 = ArrayCatalog({'Position': halo_cat2})
mesh_cat = array_cat.to_mesh(Nmesh=box.N, BoxSize=(box.Lx, box.Ly, box.Lz),
                             window='tsc', compensated=True)
#mesh_cat2 = array_cat2.to_mesh(Nmesh=box.N, BoxSize=(box.Lx, box.Ly, box.Lz),
#                               window='tsc', compensated=True)

# Put density field onto mesh
mesh_delta = ArrayMesh(box.delta_x, BoxSize=(box.Lx, box.Ly, box.Lz))
#mesh_delta = ArrayMesh(delta_ln, BoxSize=(box.Lx, box.Ly, box.Lz))

# Calculate power spectra
pspec1 = FFTPower(first=mesh_cat, mode='1d', second=None, los=[0, 0, 1], Nmu=5)
pspec1.run()
#pspec1a = FFTPower(first=mesh_cat, mode='1d', second=mesh_cat2, los=[0, 0, 1], Nmu=5)
#pspec1a.run()
pspec2 = FFTPower(first=mesh_delta, mode='1d', second=None, los=[0, 0, 1], Nmu=5)
pspec2.run()
pspec3 = FFTPower(first=mesh_delta, mode='1d', second=mesh_cat, los=[0, 0, 1], Nmu=5)
pspec3.run()

print("-"*50)
print("Halo-Halo")
print("-"*50)
for k in pspec1.attrs:
    print("%s = %s" %(k, str(pspec1.attrs[k])))

print("-"*50)
print("Delta-Delta")
print("-"*50)
for k in pspec2.attrs:
    print("%s = %s" %(k, str(pspec2.attrs[k])))

# Plot power spectra
plt.subplot(111)
plt.plot(pspec1.power['k'], pspec1.power['power'].real, 'r.', 
         ls='solid', label="hh")
#plt.plot(pspec1a.power['k'], pspec1a.power['power'].real, 'g.', 
#         ls='dashed', label="hh*")
plt.plot(pspec2.power['k'], pspec2.power['power'].real, 'bx', 
         ls='solid', label="$\delta\delta$ lin")
plt.plot(pspec3.power['k'], pspec3.power['power'].real, 'y+', 
         ls='solid', label="$h\delta$")
plt.yscale('log')
plt.ylim((1e2, 1e5))
plt.legend(loc='upper right', frameon=False)

plt.show()

exit()


plt.figure()
plt.subplot(111)
plt.matshow(box.delta_x[0,:,:], vmin=-1., vmax=5., cmap='cividis', fignum=False)
plt.title("Density field (smoothed)")
plt.colorbar()

idxs = np.where(np.logical_and(halo_cat[:,2] >= 0., halo_cat[:,2] < 2.*box.Lz/box.N))
_x, _y, _z = halo_cat[idxs].T
plt.plot(_x*box.N/box.Lx, _y*box.N/box.Ly, marker='.', color='r', ms=3., ls='none')

print("Min/max:", np.min(halo_cat), np.max(halo_cat))

#plt.matshow(Nh[0,:,:], cmap='cividis')
plt.title("Halo count")

plt.show()
