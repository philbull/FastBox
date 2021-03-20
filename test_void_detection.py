#!/usr/bin/env python

import numpy as np
import pylab as plt
from fastbox.box import CosmoBox, default_cosmo
from numpy import fft
from skimage.segmentation import watershed
import time

# Gaussian box
np.random.seed(10)
box = CosmoBox(cosmo=default_cosmo, box_scale=(1e3, 1e3, 1e3), nsamp=64, realise_now=False)
box.realise_density()
box.realise_velocity()

# Plot real-space density field 
#plt.matshow(box.delta_x[0], vmin=-1., vmax=20., cmap='cividis')
#plt.title("Real-space")
#plt.colorbar()

# Get redshift-space density field
vel_z = fft.ifftn(box.velocity_k[2]).real

delta_s = box.redshift_space_density(delta_x=box.delta_x, velocity_z=vel_z, 
                                     sigma_nl=200., method='linear')

plt.matshow(delta_s[40], vmin=-1., vmax=2., cmap='cividis')
plt.title("Redshift-space")
plt.colorbar()

#plt.show()



# Apply watershed algorithm to find voids
print("Running watershed algorithm")
t0 = time.time()
region_labels = watershed(delta_s, markers=None)
print("Watershed took %2.2f sec" % (time.time() - t0))
print("No. regions:", np.unique(region_labels).size)
region_labels = region_labels.astype(np.float64)

regions = np.unique(region_labels)
avg_delta = np.zeros(regions.shape)
accepted_regions = []
for i, r in enumerate(regions):
    if i % 100 == 0: print("    Region", i)
    avg_delta[i] = np.mean( delta_s[np.where(region_labels==r)] )
    
    
    if avg_delta[i] > -0.1:
        region_labels[np.where(region_labels==r)] = np.inf
    else:
        accepted_regions.append(r)
print("No. regions kept:", len(accepted_regions))

plt.matshow(region_labels[40])
plt.title("Voids")
plt.colorbar()

plt.show()
