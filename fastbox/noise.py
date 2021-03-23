"""
Add foreground emission to datacubes.
"""
import numpy as np
import pyccl as ccl
import pylab as plt
from numpy import fft
import scipy.ndimage


class NoiseModel(object):
    
    def __init__(self, box):
        """
        An object to manage the addition of noise on top of a realisation 
        of a density field in a box.
        
        Parameters
        ----------
        box : CosmoBox
            Object containing a simulation box.
        """
        self.box = box
    
    
    def realise_radiometer_noise(self, Tinst, tp, fov, Ndish, redshift=None):
        """
        Realisation of white noise based on the radiometer equation.
        The instrument temperature is added to a frequency-dependent sky 
        temperature, T_sky ~ 60 K (nu / 300 MHz)^-2.5. 
        
        Parameters
        ----------
        Tinst : float
            Instrument temperature, in Kelvin. This is added to the sky 
            temperature to form the total system temperature.
        
        tp : float
            Total integration time per pointing, in hours.
        
        fov : float
            Field of view (i.e. solid angle per pointing) in deg^2.
        
        Ndish : int
            Number of dishes/receivers combined to reach effective noise level.
        
        redshift : float, optional
            Redshift to evaluate the centre of the box at. Default: Same value 
            as self.box.redshift.
        
        Returns
        -------
        noise : array_like
            Realisation of noise in datacube. Units in mK.
        """
        # Get frequency array and channel width (MHz)
        freqs = self.box.freq_array(redshift=redshift)
        dnu = np.abs(freqs[1] - freqs[0])
        
        # Convert time per pointing to seconds
        tp *= 3600. # hrs to sec
        
        # Calculate time per angular resolution element (i.e. per pixel)
        ang_x, ang_y = self.box.pixel_array(redshift=redshift)
        dtheta = ang_x[1] - ang_x[0] # pixel angular size, degrees
        t_res = tp * dtheta**2. / fov # fov also in deg^2
        
        # Get Tsys as a function of frequency
        Tsky = 60e3 * (freqs / 300.)**(-2.5) # Kelvin
        Tsys = Tinst + Tsky
        
        # Noise rms from the radiometer equation (fn. of frequency only)
        sigma_rms = Tsys / np.sqrt(Ndish * t_res * (dnu * 1e6))
        
        # Generate unit white noise and multiply by noise rms
        noise = np.random.normal(0., 1., self.box.Kx.shape)
        noise *= sigma_rms[np.newaxis,np.newaxis,:]
        return noise
        
        
        
