#!/usr/bin/env python

import numpy as np
import pylab as plt
from fastbox.box import CosmoBox, default_cosmo
from fastbox.foregrounds import ForegroundModel

# Gaussian box
np.random.seed(10)
box = CosmoBox(cosmo=default_cosmo, box_scale=(3e3,3e3,1e3), 
               nsamp=64, realise_now=False)
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
plt.matshow(alpha)
plt.colorbar()
#plt.show()


# Construct foreground datacube    
cube = fg.construct_cube(fg_map, alpha, freq_ref=130., redshift=0.4)
plt.matshow(cube[:,:,0])
plt.title("FG cube")
plt.colorbar()
plt.show()
