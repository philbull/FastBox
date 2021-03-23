"""
Add foreground emission to datacubes.
"""
import numpy as np
import pyccl as ccl
import pylab as plt
from numpy import fft
import scipy.ndimage


class ForegroundModel(object):
    
    def __init__(self, box):
        """
        An object to manage the addition of foregrounds on top of a realisation 
        of a density field in a box.
        
        Parameters
        ----------
        box : CosmoBox
            Object containing a simulation box.
        """
        self.box = box
        
    
    def realise_foreground_amp(self, amp, beta, monopole, smoothing_scale=None, 
                               redshift=None):
        """
        Create realisation of the matter power spectrum by randomly sampling 
        from Gaussian distributions of variance P(k) for each k mode.
        
        Parameters
        ----------
        amp : float
            Amplitude of foreground power spectrum, in units of the field 
            squared (e.g. if the field is in units of mK, this should be in 
            [mK]^2).
        
        beta : float
            Angular power-law index.
        
        monopole : float
            Zero-point offset of the foreground at the reference frequency. 
            Should be in the same units as the field, e.g. mK.
        
        smoothing_scale : float, optional
            Additional angular smoothing scale, in degrees. Default: None 
            (no smoothing). 
            
        redshift : float, optional
            Redshift to evaluate the centre of the box at. Default: Same value 
            as self.box.redshift.
        """
        # Check redshift
        if redshift is None:
            redshift = self.box.redshift
        scale_factor = 1. / (1. + redshift)
        
        # Calculate comoving distance to box redshift
        r = ccl.comoving_angular_distance(self.box.cosmo, scale_factor)
        
        # Angular Fourier modes (in 2D)
        k_perp = 2.*np.pi * np.sqrt(  (self.box.Kx[:,:,0]/self.box.Lx)**2. 
                                    + (self.box.Ky[:,:,0]/self.box.Ly)**2.)
        
        # Foreground angular power spectrum
        # \ell ~ k_perp r / 2
        # Use FG power spectrum model from Santos et al. (2005)
        C_ell = amp * (0.5*k_perp*r / 1000.)**(beta)
        C_ell[np.isinf(C_ell)] = 0. # Remove inf at k=0
        
        # Normalise the power spectrum properly (factor of area, and norm. 
        # factor of 2D DFT)
        C_ell *= (self.box.N**4.) / (self.box.Lx * self.box.Ly)
        
        # Generate Gaussian random field with given power spectrum
        re = np.random.normal(0.0, 1.0, np.shape(k_perp))
        im = np.random.normal(0.0, 1.0, np.shape(k_perp))
        fg_k = (re + 1.j*im) * np.sqrt(C_ell) # factor of 2x larger variance
        fg_k[k_perp==0.] = 0. # remove zero mode
        
        # Transform to real space. Discarding the imag part fixes the extra 
        # factor of 2x in variance from above
        fg_x = fft.ifftn(fg_k).real + monopole
        
        # Apply angular smoothing
        if smoothing_scale is not None:
            ang_x, ang_y = self.box.pixel_array(redshift=redshift)
            sigma = smoothing_scale / (ang_x[1] - ang_x[0])
            fg_x = scipy.ndimage.gaussian_filter(fg_x, sigma=sigma, mode='wrap')
        
        return fg_x
    
    
    def realise_spectral_index(self, mean_spec_idx, std_spec_idx, 
                               smoothing_scale, redshift=None):
        """
        Generate a Gaussian random realisation of the spectral index and apply 
        a smoothing scale.
        
        Parameters
        ----------
        mean_spec_idx : float
            Mean value of the spectral index.
        
        std_spec_idx : float
            Standard deviation of the spectral index.
        
        smoothing_scale : float
            Angular smoothing scale, in degrees.
        
        redshift : float, optional
            Redshift to evaluate the centre of the box at. Default: Same value 
            as self.box.redshift.
        """
        # Generate uncorrelated Gaussian random field
        alpha = np.random.normal(mean_spec_idx, std_spec_idx, 
                                 self.box.Kx[:,:,0].shape)
        
        # Smooth with isotropic Gaussian
        ang_x, ang_y = self.box.pixel_array(redshift=redshift)
        sigma = smoothing_scale / (ang_x[1] - ang_x[0])
        alpha = scipy.ndimage.gaussian_filter(alpha, sigma=sigma, mode='wrap')
        return alpha
    
    
    def construct_cube(self, amps, spectral_idx, freq_ref=130., redshift=None):
        """
        Construct a foreground datacube from an input 2D amplitude map and 
        spectra index map.
        
        Parameters
        ----------
        amps : array_like
            2D array of amplitudes.
        
        spectral_index : array_like
            2D array of spectral indices, or float.
        
        freq_ref : float, optional
            Reference frequency, in MHz. Default: 130.
        
        redshift : float, optional
            Redshift to evaluate the centre of the box at. Default: Same value 
            as self.box.redshift.
        """
        # Get frequency array and scaling
        freqs = self.box.freq_array(redshift=redshift)
        if isinstance(spectral_idx, float):
            ffac = ((freqs / freq_ref)**spectral_idx)[np.newaxis,np.newaxis,:]
        else:
            ffac = (freqs/freq_ref)[np.newaxis,np.newaxis,:]**spectral_idx[:,:,np.newaxis]
        
        # Return datacube
        return amps[:,:,np.newaxis] * ffac
        
    
