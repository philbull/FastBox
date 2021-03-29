"""
Classes to handle instrumental beams.
"""
import numpy as np
import pyccl as ccl
import pylab as plt
from numpy import fft
import scipy.ndimage
from scipy.signal import convolve2d
#import katbeam


class BeamModel(object):
    
    def __init__(self, box):
        """
        An object to manage a beam model as a function of angle and frequency.
        
        Parameters  
        ----------
        box : CosmoBox
            Object containing a simulation box.
        """
        self.box = box
    
    
    def beam_cube(self, pol=None):
        """
        Return beam values in a cube matching the shape of the box.
        
        Returns
        -------
        beam : array_like
            Beam value at each voxel in `self.box`.
        """
        return np.ones((self.box.N, self.box.N, self.box.N))
    
    
    def beam_value(self, x, y, freq, pol=None):
        """
        Return the beam value at a particular set of coordinates.
        
        The x, y, and freq arrays should have the same length.
        
        Parameters
        ----------
        x, y : array_like
            Angular position in degrees.
        
        freq : array_like
            Frequency (in MHz).
        
        Returns
        -------
        beam : array_like
            Value of the beam at the specified coordinates.
        """
        assert x.shape == y.shape == freq.shape, \
            "x, y, and freq arrays should have the same shape"
        return 1. + 0.*x
    
    
    def convolve_real(self, field_x, pol=None):
        """
        Perform a real-space (direct) convolution of a field with the beam. 
        Each frequency channel is convolved separately.
        
        Parameters
        ----------
        field_x : array_like
            Field to be convolved with the beam. Must be a 3D array; the freq. 
            direction is assumed to be the last one.
        
        pol : str
            Which polarisation to return the beam for. Default: None.
        
        Returns
        -------
        field_smoothed : array_like
            Beam-convolved field, same shape as the input field.
        """
        beam = self.beam_cube(pol=pol)
        field_sm = np.zeros_like(field_x)
        for i in range(field_x.shape[-1]):
            field_sm[:,:,i] = convolve2d(field_x[:,:,i], beam[:,:,i], 
                                         mode='same', boundary='wrap', 
                                         fillvalue=0.)
        return field_sm



class KatBeamModel(BeamModel):

    def __init__(self, box, model='L'):
        """
        An object to manage a beam based on the KatBeam "JimBeam" model as a 
        function of angle and frequency.
        
        Parameters  
        ----------
        box : CosmoBox
            Object containing a simulation box.
        
        model : str, optional
            Which model to use from katbeam.JimBeam. Options are 'L' (L-band), 
            or 'UHF' (UHF-band).
        """
        # Try to import katbeam
        try:
            import katbeam
        except:
            raise ImportError("Unable to import `katbeam`; please install from "
                              "https://github.com/ska-sa/katbeam")
        
        # Save box
        self.box = box
        
        # List of available models
        self.avail_models = { 'L':   'MKAT-AA-L-JIM-2020',
                              'UHF': 'MKAT-AA-UHF-JIM-2020' }
        if model not in self.avail_models.keys():
            raise ValueError( "model '%s' not found. Options are: %s" 
                              % (model, list(self.avail_models.keys())) )
        self.model = model
        
        # Instantiate beam object
        self.beam = katbeam.JimBeam(self.avail_models[model])
    
    
    def beam_cube(self, pol='I'):
        """
        Return beam values in a cube matching the shape of the box.
        
        Returns
        -------
        beam : array_like
            Beam value at each voxel in `self.box`.
        
        pol : str, optional
            Which polarisation to return the beam for. Options are 'I', 'HH', 
            and 'VV'. Default: 'I'.
        """
        assert pol in ['I', 'HH', 'VV'], "Unknown polarisation '%s'" % pol
        
        # Get pixel and frequency arrays and expand into meshes
        ang_x, ang_y = self.box.pixel_array() # in degrees
        freqs = self.box.freq_array() # in MHz
        x, y, nu = np.meshgrid(ang_x, ang_y, freqs)
        
        # Return beam interpolated onto grid for chosen polarisation
        if pol == 'HH':
            return self.beam.HH(x, y, nu)
        elif pol == 'VV':
            return self.beam.VV(x, y, nu)
        else:
            return self.beam.I(x, y, nu)
            
    
    def beam_value(self, x, y, freq, pol='I'):
        """
        Return the beam value at a particular set of coordinates.
        
        The x, y, and freq arrays should have the same length.
        
        Parameters
        ----------
        x, y : array_like
            Angular position in degrees.
        
        freq : array_like
            Frequency (in MHz).
        
        pol : str, optional
            Which polarisation to return the beam for. Options are 'I', 'HH', 
            and 'VV'. Default: 'I'.
        
        Returns
        -------
        beam : array_like
            Value of the beam at the specified coordinates.
        """
        assert x.shape == y.shape == freq.shape, \
            "x, y, and freq arrays should have the same shape"
        assert pol in ['I', 'HH', 'VV'], "Unknown polarisation '%s'" % pol
        
        # Return beam interpolated at input values for chosen polarisation
        if pol == 'HH':
            return self.beam.HH(x, y, freq)
        elif pol == 'VV':
            return self.beam.VV(x, y, freq)
        else:
            return self.beam.I(x, y, freq)

