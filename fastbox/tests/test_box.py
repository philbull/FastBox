#!/usr/bin/env python

import pytest
import numpy as np
from fastbox.box import CosmoBox, default_cosmo

def test_gaussian_box():
    """Generate Gaussian density field in box."""
    # Realise Gaussian box
    np.random.seed(11)
    box = CosmoBox(cosmo=default_cosmo, box_scale=(1e2, 1e2, 1e2), nsamp=16, 
                   realise_now=False)
    box.realise_density()
    
    # Check that density field is valid
    assert box.delta_x.shape == (16, 16, 16)
    assert box.delta_x.dtype == np.float64
    assert np.all(~np.isnan(box.delta_x))
    
    # Realise density field with same random seed and realise_now=True, and 
    # manually setting the redshift and a single box_scale
    np.random.seed(11)
    box2 = CosmoBox(cosmo=default_cosmo, box_scale=1e2, nsamp=16, 
                    redshift=0., realise_now=True)
    
    assert np.allclose(box.delta_x, box2.delta_x)
    
    # Check that pixel resolution etc. is correct
    assert box.Lx == box.Ly == box.Lz == 1e2
    assert box.x.size == box.y.size == box.z.size == 16
    assert np.isclose(np.max(box.x) - np.min(box.x), 1e2)
    
    # Check that cuboidal boxes work
    box3 = CosmoBox(cosmo=default_cosmo, box_scale=(1e2, 2e2, 1e3), nsamp=16, 
                    redshift=1., realise_now=True)
    assert box3.delta_x.shape == (16, 16, 16)
    assert box3.delta_x.dtype == np.float64
    assert np.all(~np.isnan(box3.delta_x))


def test_lognormal_box():
    """Generate log-normal density field in box."""
    # Realise Gaussian box
    np.random.seed(11)
    box = CosmoBox(cosmo=default_cosmo, box_scale=(1e2, 1e2, 1e2), nsamp=16, 
                   realise_now=True)
    
    # Apply log-normal transform
    delta_log = box.lognormal(box.delta_x)
    
    # Check that log-normal density field is valid
    assert delta_log.shape == (16, 16, 16)
    # assert delta_log.dtype == np.float64
    assert np.all(~np.isnan(delta_log))
    assert np.all(delta_log >= -1.) # delta_log >= -1
    

def test_box_redshift_space_density():
    """Check that a redshift-space density field can be generated."""
    
    # Realise Gaussian box and velocity field
    np.random.seed(11)
    box = CosmoBox(cosmo=default_cosmo, box_scale=(1e2, 1e2, 1e2), nsamp=16, 
                   realise_now=False)
    box.realise_density()
    box.realise_velocity()
    
    # Get redshift-space density field
    vel_z = np.fft.ifftn(box.velocity_k[2]).real
    delta_s = box.redshift_space_density(delta_x=box.delta_x, velocity_z=vel_z, 
                                         sigma_nl=200., method='linear')
    
    # Check that redshift-space density field is valid
    assert delta_s.shape == (16, 16, 16)
    # assert delta_s.dtype == np.float64
    assert np.all(~np.isnan(delta_s))


def test_box_transfer_function():
    """Check that a transfer function can be applied to the density field."""
    
    # Realise Gaussian box
    np.random.seed(11)
    box = CosmoBox(cosmo=default_cosmo, box_scale=(1e2, 1e2, 1e2), nsamp=16, 
                   realise_now=True)
                   
    # Gaussian box with beam smoothing and foreground cut
    transfer_fn = lambda k_perp, k_par: \
        (1. - np.exp(-0.5 * (k_par/0.001)**2.)) \
        * np.exp(-0.5 * (k_perp/0.1)**2.)
    delta_smoothed = box.apply_transfer_fn(box.delta_k, transfer_fn=transfer_fn)
    
    # Check that smoothed density field is valid
    assert delta_smoothed.shape == (16, 16, 16)
    # assert delta_smoothed.dtype == np.float64
    assert np.all(~np.isnan(delta_smoothed))


def test_box_power_spectrum():
    """Check that the theoretical and box power spectra can be calculated."""
    
    # Realise Gaussian box
    np.random.seed(14)
    box = CosmoBox(cosmo=default_cosmo, box_scale=(1e3, 1e3, 1e3), nsamp=64, 
                   realise_now=False)
    box.realise_density()
    
    # Calculate binned power spectrum and theoretical power spectrum
    re_k, re_pk, re_stddev = box.binned_power_spectrum()
    th_k, th_pk = box.theoretical_power_spectrum()
    
    # Check that sigma(R) and sigma_8 can be calculated
    sigR = box.sigmaR(R=8.) # R in units of Mpc/h
    sig8 = box.sigma8()
    assert np.isclose(sigR, sig8)
    
    # Run built-in test to print a report on sampling accuracy
    box.test_sampling_error()
    
    # Check that sigma_8 calculated from box is close to input cosmo sigma_8 
    # (this depends on box size/resolution)
    assert np.abs(sig8 - box.cosmo['sigma8']) < 0.09 # 0.09 is empirical


def test_box_coordinates():
    """Check that pixel and frequency coordinates are returned."""
    
    # Realise Gaussian box
    np.random.seed(22)
    box = CosmoBox(cosmo=default_cosmo, box_scale=(1e3, 1e3, 1e3), nsamp=16, 
                   realise_now=True, redshift=0.8)
    
    # Check pixel array
    ang_x, ang_y = box.pixel_array()
    ang_x2, ang_y2 = box.pixel_array(redshift=0.82)
    # ^Higher z, so further away, so smaller angle
    
    # Check for valid output
    assert np.all(~np.isnan(ang_x))
    assert np.all(~np.isnan(ang_y))
    assert np.all(~np.isnan(ang_x2))
    assert np.all(~np.isnan(ang_y2))
    
    # Square box => equal pixel sizes
    assert np.isclose(ang_x[1] - ang_x[0], ang_y[1] - ang_y[0])
    
    # Check that higher redshift pixels are smaller
    assert ang_x[1] - ang_x[0] > ang_x2[1] - ang_x2[0]
    assert ang_y[1] - ang_y[0] > ang_y2[1] - ang_y2[0]
    
    # Check that frequency array goes in descending order (highest z coord => 
    # lowest frequency)
    assert np.all(np.diff(box.freq_array()) < 0.) # negative differences
    assert np.all(np.diff(box.freq_array(redshift=2.)) < 0.) # negative differences
    

def test_box_errors():
    """Check that correct errors are raised for invalid input."""
    
    # Invalid cosmology object passed in
    with pytest.raises(TypeError):
        box = CosmoBox(cosmo=[0.7, 0.3], box_scale=(1e2, 1e2, 1e2), nsamp=16, 
                       realise_now=False)


def test_box_builtin_tests():
    """Run the built-in tests in the CosmoBox object."""
    box = CosmoBox(cosmo=default_cosmo, box_scale=(1e2, 1e2, 1e2), nsamp=16, 
                   realise_now=True)
    
    # Test Parseval's theorem (integrals of power in real and Fourier space are 
    # equal)
    s1, s2 = box.test_parseval()
    assert np.isclose(s1, s2)


