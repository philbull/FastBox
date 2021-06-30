#!/usr/bin/env python

import pytest
import numpy as np
from fastbox.box import CosmoBox, default_cosmo

def test_gaussian_box():
    # Gaussian box
    np.random.seed(11)
    box = CosmoBox(cosmo=default_cosmo, box_scale=(1e2, 1e2, 1e2), nsamp=32, 
                   realise_now=False)
    box.realise_density()
    
    assert box.delta_x.shape == (32, 32, 32)
    assert box.delta_x.dtype == np.float64
    assert np.all(~np.isnan(box.delta_x))
    
    #re_k, re_pk, re_stddev = box.binned_power_spectrum()
    #th_k, th_pk = box.theoretical_power_spectrum()

