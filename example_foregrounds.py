#!/usr/bin/env python

import numpy as np
import pylab as plt
from fastbox.box import CosmoBox, default_cosmo
from fastbox.foregrounds import ForegroundModel
import fastbox 

# Gaussian box
np.random.seed(10)
box = CosmoBox(cosmo=default_cosmo, box_scale=(3e3,3e3,1e3), 
               nsamp=32, realise_now=False)
#box.realise_density()

# Foreground model
fg = ForegroundModel(box)

fg_map = fg.realise_foreground_amp(amp=57., beta=1.1, monopole=10., 
                                   redshift=0.4)
#plt.matshow(fg_map)
#plt.colorbar()
#plt.show()

# Construct spectral index map
ang_x, ang_y = box.pixel_array(redshift=0.4)
print("Pixel size:", ang_x[1] - ang_x[0], "deg.")

alpha = fg.realise_spectral_index(mean_spec_idx=2.07, std_spec_idx=0.2,
                                  smoothing_scale=15., redshift=0.4)
#plt.matshow(alpha)
#plt.colorbar()
#plt.show()

# Construct foreground datacube
fgcube = fg.construct_cube(fg_map, alpha, freq_ref=130., redshift=0.4)


# Apply PCA cleaning
cleaned_cube, U_fg, amp_fg = fastbox.filters.pca_filter(fgcube, 
                                                        nmodes=3, 
                                                        return_filter=True)

print(np.mean(cleaned_cube))
print(amp_fg.shape)

plt.plot(amp_fg.T)


# Plot FG cube
plt.matshow(fgcube[:,0,:].real)
plt.title("FG cube")
plt.colorbar()

# Plot cleaned cube
plt.matshow(cleaned_cube[:,0,:].real)
plt.title("FG-cleaned cube")
plt.colorbar()

plt.show()

