"""
Classes to handle a biased tracer on top of a density field.
"""
import numpy as np
import pyccl as ccl
import pylab as plt
from numpy import fft
import scipy.ndimage


class TracerModel(object):
    
    def __init__(self, box):
        """
        An object to manage a biased tracer on top of a realisation of a 
        density field in a box.
        
        Parameters
        ----------
        box : CosmoBox
            Object containing a simulation box.
        """
        self.box = box
    
    
    def signal_amplitude(self, amp, redshift):
        """
        Overall signal amplitude (e.g. mean brightness temperature). This is a 
        simple constant-amplitude model.
        
        Parameters
        ----------
        amp : float
            Overall amplitude.
        
        redshift : float
            Redshift to evaluate the amplitude at.
        
        Returns
        -------
        bias : float
            Bias at given redshift.
        """
        return amp + 0.*redshift # same shape as redshift
    
    
    def linear_bias(self, b0, redshift):
        """
        Linear bias model, b(z) = b0 sqrt(1 + z)
        
        Parameters
        ----------
        b0 : float
            Overall bias amplitude.
        
        redshift : float
            Redshift to evaluate the bias at.
        
        Returns
        -------
        bias : float
            Bias at given redshift.
        """
        return b0 * np.sqrt(1. + redshift)



class HITracer(TracerModel):
    
    def __init__(self, box, OmegaHI0=0.000486, bHI0):
        """
        An object to manage a biased tracer on top of a realisation of a 
        density field in a box.
        
        Parameters
        ----------
        box : CosmoBox
            Object containing a simulation box.
        
        OmegaHI0 : float, optional
            Fractional density of HI at redshift 0. Default: 0.000486.
            
        bHI0 : float, optional
            HI bias at redshift 0. Default: 
        """
        # Initialise superclass
        super().__init__(box)
        
        # Set parameters
        self.OmegaHI0 = OmegaHI0
        self.bHI0 = bHI0
        
        
    def signal_amplitude(self, redshift=None, formula='powerlaw'):
        """
        Brightness temperature Tb(z), in mK. Several different expressions for the 
        21cm line brightness temperature are available:
        
        Parameters
        ----------
        redshift : float, optional
            Central redshift to evaluate the signal amplitude at. If not 
            specified, uses `self.box.redshift`. Default: None.
        
        formula : str, optional
            Which fitting formula to use for the brightness temperature. Some 
            of the options are a function of Omega_HI(z)
              - 'powerlaw': Simple power-law fit to Mario's updated data 
                (powerlaw M_HI function with alpha=0.6) (Default)
              - 'hall': From Hall, Bonvin, and Challinor.
           Default: 'powerlaw'
        """
        if redshift is None:
            redshift = self.box.redshift
        z = redshift
        
        # Calculate OmegaHI(z)
        omegaHI = self.Omega_HI(redshift=redshift)
        
        # Select which formula to use
        if formula == 'powerlaw':
            # Mario Santos' fit, used in Bull et al. (2015)
            Tb = 5.5919e-02 + 2.3242e-01*z - 2.4136e-02*z**2.
            
        elif formula == 'hall':
            # From Hall et al.
            E = ccl.h_over_h0(self.box.cosmo, 1./(1.+z))
            Tb = 188. * self.box.cosmo['h'] * omegaHI * (1.+z)**2. / E
            
        else:
            raise ValueError("No formula found with name '%s'" % formula)
        return Tb
    
    
    def bias_HI(self, redshift=None):
        """
        HI bias as a function of redshift.
        
        Parameters
        ----------
        redshift : float, optional
            Central redshift to evaluate the signal amplitude at. If not 
            specified, uses `self.box.redshift`. Default: None.
        """
        if redshift is None:
            redshift = self.box.redshift
        z = redshift
        
        # Fitting formula, based on Mario Santos' halo model calculation 
        # (see Bull et al. 2015)
        return (self.bHI0/0.677105)*(6.6655e-01 + 1.7765e-01*z + 5.0223e-02*z**2.)
    
    
    def Omega_HI(self, redshift=None, formula='powerlaw'):
        """
        Fractional density of HI as a function of redshift, from a fitting 
        function.
        
        Parameters
        ----------
        Default: 'powerlaw'
        """
        if redshift is None:
            redshift = self.box.redshift
        z = redshift
        
        # Fitting formula; see Bull et al. (2015)
        return (self.Omega_HI0 / 0.000486) \
             * (4.8304e-04 + 3.8856e-04*z - 6.5119e-05*z**2.)

