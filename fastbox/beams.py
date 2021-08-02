"""
Classes to handle instrumental beams.
"""
import numpy as np
import pyccl as ccl
import pylab as plt
from numpy import fft
import scipy.ndimage
from scipy.signal import convolve2d, fftconvolve
from multiprocessing import Pool


class BeamModel(object):
    
    def __init__(self, box):
        """
        An object to manage a beam model as a function of angle and frequency.
        
        Parameters:
            box (CosmoBox):
                Object containing a simulation box.
        """
        self.box = box
    
    
    def beam_cube(self, pol=None):
        """
        Return beam values in a cube matching the shape of the box.
        
        Parameters:
            pol (str):
                Polarisation.
        
        Returns:
            beam (array_like):
                Beam value at each voxel in `self.box`.
        """
        return np.ones((self.box.N, self.box.N, self.box.N))
    
    
    def beam_value(self, x, y, freq, pol=None):
        """
        Return the beam value at a particular set of coordinates.
        
        The x, y, and freq arrays should have the same length.
        
        Parameters:
            x, y (array_like):
                Angular position in degrees.
            
            freq (array_like):
                Frequency (in MHz).
        
        Returns:
            beam (array_like):
                Value of the beam at the specified coordinates.
        """
        assert x.shape == y.shape == freq.shape, \
            "x, y, and freq arrays should have the same shape"
        return 1. + 0.*x
    
    
    def convolve_fft(self, field_x, pol=None):
        """
        Perform an FFT-based convolution of a field with the beam. Each 
        frequency channel is convolved separately.
        
        Parameters:
            field_x (array_like):
                Field to be convolved with the beam. Must be a 3D array; the freq. 
                direction is assumed to be the last one.
            
            pol (str, optional):
                Which polarisation to return the beam for.
        
        Returns:
            field_smoothed (array_like):
                Beam-convolved field, same shape as the input field.
        """
        # Get beam cube and normalise (so integral is unity)
        beam = self.beam_cube(pol=pol)
        norm = np.sum(beam.reshape(-1, beam.shape[-1]), axis=0)
        
        # Do 2D FFT convolution with beam on each frequency slice
        field_sm = scipy.signal.fftconvolve(beam, field_x, mode='same', 
                                            axes=[0,1])
        return field_sm / norm[np.newaxis,np.newaxis,:]

    
    def convolve_real(self, field_x, pol=None, verbose=False):
        """
        Perform a real-space (direct) convolution of a field with the beam. 
        Each frequency channel is convolved separately. This function can take 
        a long time.
        
        Parameters:
            field_x (array_like):
                Field to be convolved with the beam. Must be a 3D array; the freq. 
                direction is assumed to be the last one.
            
            pol (str, optional):
                Which polarisation to return the beam for.
            
            verbose (bool, optional):
                Whether to print progress messages.
        
        Returns:
            field_smoothed (array_like):
                Beam-convolved field, same shape as the input field.
        """
        # Get beam cube and normalise (so integral is unity)
        beam = self.beam_cube(pol=pol)
        norm = np.sum(beam.reshape(-1, beam.shape[-1]), axis=0)
        
        # Initialise smoothed field
        field_sm = np.zeros_like(field_x)
        
        # Function to do beam convolution of a single frequency slice 
        #def conv(i):
        #    return convolve2d(field_x[:,:,i], beam[:,:,i], 
        #                      mode='same', boundary='wrap', fillvalue=0.)
        
        # Loop over frequency slices (in parallel if possible)
        #with Pool(nproc) as pool:
        #    field_sm = np.array( pool.map(conv, np.arange(field_x.shape[-1])) )
        
        # Loop over frequencies
        for i in range(field_x.shape[-1]):
            if verbose and i % 10 == 0:
                print("convolve_real: %d / %d" % (i+1, field_x.shape[-1]))
                
            field_sm[:,:,i] = convolve2d(beam[:,:,i], field_x[:,:,i], 
                                         mode='same', boundary='wrap', 
                                         fillvalue=0.)
        return field_sm / norm[np.newaxis,np.newaxis,:]



class KatBeamModel(BeamModel):

    def __init__(self, box, model='L'):
        """
        An object to manage a beam based on the KatBeam "JimBeam" model as a 
        function of angle and frequency.
        
        Parameters:
            box (CosmoBox):
                Object containing a simulation box.
        
            model (str, optional):
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
        
        Parameters:
            pol (str, optional):
                Which polarisation to return the beam for. Options are 'I', 'HH', 
                and 'VV'.
            
        Returns:
            beam (array_like):
                Beam value at each voxel in `self.box`.
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
        
        Parameters:
            x, y (array_like):
                Angular position in degrees.
            
            freq (array_like):
                Frequency (in MHz).
            
            pol (str, optional):
                Which polarisation to return the beam for. Options are 'I', 'HH', 
                and 'VV'.
        
        Returns:
            beam (array_like):
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


class ZernikeBeamModel(BeamModel):
    
    def __init__(self, box, coeffs):
        """
        An object to manage a beam based on a Zernike polynomial expansion.
        
        Parameters:
            box (CosmoBox):
                Object containing a simulation box.
        
        coeffs (array_like):
            Zernike polynomial coefficients.
        """
        self.box = box
        self.coeffs = coeffs
    
    
    def beam_cube(self):
        """
        Return beam values in a cube matching the shape of the box.
        
        Returns:
            beam (array_like):
                Beam value at each voxel in `self.box`.
        """
        assert pol in ['I', 'HH', 'VV'], "Unknown polarisation '%s'" % pol
        
        # Get pixel and frequency arrays and expand into meshes
        ang_x, ang_y = self.box.pixel_array() # in degrees
        freqs = self.box.freq_array() # in MHz
        x, y, nu = np.meshgrid(ang_x, ang_y, freqs)
        
        # Convert x and y to angle cosines
        xcos = np.sin(x * np.pi/180.)
        ycos = np.sin(y * np.pi/180.)
        
        # Return beam interpolated onto grid
        return self.zernike(self.coeffs, xcos, ycos)
            
    
    def beam_value(self, x, y, freq):
        """
        Return the beam value at a particular set of coordinates. The beam 
        centre is intended to be at x = y = 0 degrees.
        
        The x, y, and freq arrays should have the same length.
        
        Parameters:
            x, y (array_like):
                Angular position in degrees.
            
            freq (array_like):
                Frequency (in MHz).
        
        Returns:
            beam (array_like):
                Value of the beam at the specified coordinates.
        """
        assert x.shape == y.shape == freq.shape, \
            "x, y, and freq arrays should have the same shape"
        
        # Convert x and y to angle cosines
        xcos = np.sin(x * np.pi/180.)
        ycos = np.sin(y * np.pi/180.)
        
        # Return beam calculated at input values
        return self.zernike(self.coeffs, xcos, ycos)
    
    
    def zernike(self, coeffs, x, y):
        """
        Zernike polynomials (up to degree 66) on the unit disc.

        This code was adapted from:
        https://gitlab.nrao.edu/pjaganna/zcpb/-/blob/master/zernikeAperture.py

        Parameters:
            coeffs (array_like):
                Array of real coefficients of the Zernike polynomials, from 0..66.

            x, y (array_like):
                Points on the unit disc.

        Returns:
            zernike (array_like):
                Values of the Zernike polynomial at the input x,y points.
        """
        # Coefficients
        assert len(coeffs) <= 66, "Max. number of coeffs is 66."
        c = np.zeros(66)
        c[: len(coeffs)] += coeffs

        # x2 = self.powl(x, 2)
        # y2 = self.powl(y, 2)
        x2, x3, x4, x5, x6, x7, x8, x9, x10 = (
            x ** 2.0,
            x ** 3.0,
            x ** 4.0,
            x ** 5.0,
            x ** 6.0,
            x ** 7.0,
            x ** 8.0,
            x ** 9.0,
            x ** 10.0,
        )
        y2, y3, y4, y5, y6, y7, y8, y9, y10 = (
            y ** 2.0,
            y ** 3.0,
            y ** 4.0,
            y ** 5.0,
            y ** 6.0,
            y ** 7.0,
            y ** 8.0,
            y ** 9.0,
            y ** 10.0,
        )

        # Setting the equations for the Zernike polynomials
        # r = np.sqrt(powl(x,2) + powl(y,2))
        Z1 = c[0] * 1  # m = 0    n = 0
        Z2 = c[1] * x  # m = -1   n = 1
        Z3 = c[2] * y  # m = 1    n = 1
        Z4 = c[3] * 2 * x * y  # m = -2   n = 2
        Z5 = c[4] * (2 * x2 + 2 * y2 - 1)  # m = 0  n = 2
        Z6 = c[5] * (-1 * x2 + y2)  # m = 2  n = 2
        Z7 = c[6] * (-1 * x3 + 3 * x * y2)  # m = -3     n = 3
        Z8 = c[7] * (-2 * x + 3 * (x3) + 3 * x * (y2))  # m = -1   n = 3
        Z9 = c[8] * (-2 * y + 3 * y3 + 3 * (x2) * y)  # m = 1    n = 3
        Z10 = c[9] * (y3 - 3 * (x2) * y)  # m = 3 n =3
        Z11 = c[10] * (-4 * (x3) * y + 4 * x * (y3))  # m = -4    n = 4
        Z12 = c[11] * (-6 * x * y + 8 * (x3) * y + 8 * x * (y3))  # m = -2   n = 4
        Z13 = c[12] * (
            1 - 6 * x2 - 6 * y2 + 6 * x4 + 12 * (x2) * (y2) + 6 * y4
        )  # m = 0  n = 4
        Z14 = c[13] * (3 * x2 - 3 * y2 - 4 * x4 + 4 * y4)  # m = 2    n = 4
        Z15 = c[14] * (x4 - 6 * (x2) * (y2) + y4)  # m = 4   n = 4
        Z16 = c[15] * (x5 - 10 * (x3) * y2 + 5 * x * (y4))  # m = -5   n = 5
        Z17 = c[16] * (
            4 * x3 - 12 * x * (y2) - 5 * x5 + 10 * (x3) * (y2) + 15 * x * y4
        )  # m =-3     n = 5
        Z18 = c[17] * (
            3 * x - 12 * x3 - 12 * x * (y2) + 10 * x5 + 20 * (x3) * (y2) + 10 * x * (y4)
        )  # m= -1  n = 5
        Z19 = c[18] * (
            3 * y - 12 * y3 - 12 * y * (x2) + 10 * y5 + 20 * (y3) * (x2) + 10 * y * (x4)
        )  # m = 1  n = 5
        Z20 = c[19] * (
            -4 * y3 + 12 * y * (x2) + 5 * y5 - 10 * (y3) * (x2) - 15 * y * x4
        )  # m = 3   n = 5
        Z21 = c[20] * (y5 - 10 * (y3) * x2 + 5 * y * (x4))  # m = 5 n = 5
        Z22 = c[21] * (6 * (x5) * y - 20 * (x3) * (y3) + 6 * x * (y5))  # m = -6 n = 6
        Z23 = c[22] * (
            20 * (x3) * y - 20 * x * (y3) - 24 * (x5) * y + 24 * x * (y5)
        )  # m = -4   n = 6
        Z24 = c[23] * (
            12 * x * y
            + 40 * (x3) * y
            - 40 * x * (y3)
            + 30 * (x5) * y
            + 60 * (x3) * (y3)
            - 30 * x * (y5)
        )  # m = -2   n = 6
        Z25 = c[24] * (
            -1
            + 12 * (x2)
            + 12 * (y2)
            - 30 * (x4)
            - 60 * (x2) * (y2)
            - 30 * (y4)
            + 20 * (x6)
            + 60 * (x4) * y2
            + 60 * (x2) * (y4)
            + 20 * (y6)
        )  # m = 0   n = 6
        Z26 = c[25] * (
            -6 * (x2)
            + 6 * (y2)
            + 20 * (x4)
            - 20 * (y4)
            - 15 * (x6)
            - 15 * (x4) * (y2)
            + 15 * (x2) * (y4)
            + 15 * (y6)
        )  # m = 2   n = 6
        Z27 = c[26] * (
            -5 * (x4)
            + 30 * (x2) * (y2)
            - 5 * (y4)
            + 6 * (x6)
            - 30 * (x4) * y2
            - 30 * (x2) * (y4)
            + 6 * (y6)
        )  # m = 4    n = 6
        Z28 = c[27] * (
            -1 * (x6) + 15 * (x4) * (y2) - 15 * (x2) * (y4) + y6
        )  # m = 6   n = 6
        Z29 = c[28] * (
            -1 * (x7) + 21 * (x5) * (y2) - 35 * (x3) * (y4) + 7 * x * (y6)
        )  # m = -7    n = 7
        Z30 = c[29] * (
            -6 * (x5)
            + 60 * (x3) * (y2)
            - 30 * x * (y4)
            + 7 * x7
            - 63 * (x5) * (y2)
            - 35 * (x3) * (y4)
            + 35 * x * (y6)
        )  # m = -5    n = 7
        Z31 = c[30] * (
            -10 * (x3)
            + 30 * x * (y2)
            + 30 * x5
            - 60 * (x3) * (y2)
            - 90 * x * (y4)
            - 21 * x7
            + 21 * (x5) * (y2)
            + 105 * (x3) * (y4)
            + 63 * x * (y6)
        )  # m =-3       n = 7
        Z32 = c[31] * (
            -4 * x
            + 30 * x3
            + 30 * x * (y2)
            - 60 * (x5)
            - 120 * (x3) * (y2)
            - 60 * x * (y4)
            + 35 * x7
            + 105 * (x5) * (y2)
            + 105 * (x3) * (y4)
            + 35 * x * (y6)
        )  # m = -1  n = 7
        Z33 = c[32] * (
            -4 * y
            + 30 * y3
            + 30 * y * (x2)
            - 60 * (y5)
            - 120 * (y3) * (x2)
            - 60 * y * (x4)
            + 35 * y7
            + 105 * (y5) * (x2)
            + 105 * (y3) * (x4)
            + 35 * y * (x6)
        )  # m = 1   n = 7
        Z34 = c[33] * (
            10 * (y3)
            - 30 * y * (x2)
            - 30 * y5
            + 60 * (y3) * (x2)
            + 90 * y * (x4)
            + 21 * y7
            - 21 * (y5) * (x2)
            - 105 * (y3) * (x4)
            - 63 * y * (x6)
        )  # m =3     n = 7
        Z35 = c[34] * (
            -6 * (y5)
            + 60 * (y3) * (x2)
            - 30 * y * (x4)
            + 7 * y7
            - 63 * (y5) * (x2)
            - 35 * (y3) * (x4)
            + 35 * y * (x6)
        )  # m = 5  n = 7
        Z36 = c[35] * (
            y7 - 21 * (y5) * (x2) + 35 * (y3) * (x4) - 7 * y * (x6)
        )  # m = 7  n = 7
        Z37 = c[36] * (
            -8 * (x7) * y + 56 * (x5) * (y3) - 56 * (x3) * (y5) + 8 * x * (y7)
        )  # m = -8  n = 8
        Z38 = c[37] * (
            -42 * (x5) * y
            + 140 * (x3) * (y3)
            - 42 * x * (y5)
            + 48 * (x7) * y
            - 112 * (x5) * (y3)
            - 112 * (x3) * (y5)
            + 48 * x * (y7)
        )  # m = -6  n = 8
        Z39 = c[38] * (
            -60 * (x3) * y
            + 60 * x * (y3)
            + 168 * (x5) * y
            - 168 * x * (y5)
            - 112 * (x7) * y
            - 112 * (x5) * (y3)
            + 112 * (x3) * (y5)
            + 112 * x * (y7)
        )  # m = -4   n = 8
        Z40 = c[39] * (
            -20 * x * y
            + 120 * (x3) * y
            + 120 * x * (y3)
            - 210 * (x5) * y
            - 420 * (x3) * (y3)
            - 210 * x * (y5)
            - 112 * (x7) * y
            + 336 * (x5) * (y3)
            + 336 * (x3) * (y5)
            + 112 * x * (y7)
        )  # m = -2   n = 8
        Z41 = c[40] * (
            1
            - 20 * x2
            - 20 * y2
            + 90 * x4
            + 180 * (x2) * (y2)
            + 90 * y4
            - 140 * x6
            - 420 * (x4) * (y2)
            - 420 * (x2) * (y4)
            - 140 * (y6)
            + 70 * x8
            + 280 * (x6) * (y2)
            + 420 * (x4) * (y4)
            + 280 * (x2) * (y6)
            + 70 * y8
        )  # m = 0    n = 8
        Z42 = c[41] * (
            10 * x2
            - 10 * y2
            - 60 * x4
            + 105 * (x4) * (y2)
            - 105 * (x2) * (y4)
            + 60 * y4
            + 105 * x6
            - 105 * y6
            - 56 * x8
            - 112 * (x6) * (y2)
            + 112 * (x2) * (y6)
            + 56 * y8
        )  # m = 2  n = 8
        Z43 = c[42] * (
            15 * x4
            - 90 * (x2) * (y2)
            + 15 * y4
            - 42 * x6
            + 210 * (x4) * (y2)
            + 210 * (x2) * (y4)
            - 42 * y6
            + 28 * x8
            - 112 * (x6) * (y2)
            - 280 * (x4) * (y4)
            - 112 * (x2) * (y6)
            + 28 * y8
        )  # m = 4     n = 8
        Z44 = c[43] * (
            7 * x6
            - 105 * (x4) * (y2)
            + 105 * (x2) * (y4)
            - 7 * y6
            - 8 * x8
            + 112 * (x6) * (y2)
            - 112 * (x2) * (y6)
            + 8 * y8
        )  # m = 6    n = 8
        Z45 = c[44] * (
            x8 - 28 * (x6) * (y2) + 70 * (x4) * (y4) - 28 * (x2) * (y6) + y8
        )  # m = 8     n = 9
        Z46 = c[45] * (
            x9 - 36 * (x7) * (y2) + 126 * (x5) * (y4) - 84 * (x3) * (y6) + 9 * x * (y8)
        )  # m = -9     n = 9
        Z47 = c[46] * (
            8 * x7
            - 168 * (x5) * (y2)
            + 280 * (x3) * (y4)
            - 56 * x * (y6)
            - 9 * x9
            + 180 * (x7) * (y2)
            - 126 * (x5) * (y4)
            - 252 * (x3) * (y6)
            + 63 * x * (y8)
        )  # m = -7    n = 9
        Z48 = c[47] * (
            21 * x5
            - 210 * (x3) * (y2)
            + 105 * x * (y4)
            - 56 * x7
            + 504 * (x5) * (y2)
            + 280 * (x3) * (y4)
            - 280 * x * (y6)
            + 36 * x9
            - 288 * (x7) * (y2)
            - 504 * (x5) * (y4)
            + 180 * x * (y8)
        )  # m = -5    n = 9
        Z49 = c[48] * (
            20 * x3
            - 60 * x * (y2)
            - 105 * x5
            + 210 * (x3) * (y2)
            + 315 * x * (y4)
            + 168 * x7
            - 168 * (x5) * (y2)
            - 840 * (x3) * (y4)
            - 504 * x * (y6)
            - 84 * x9
            + 504 * (x5) * (y4)
            + 672 * (x3) * (y6)
            + 252 * x * (y8)
        )  # m = -3  n = 9
        Z50 = c[49] * (
            5 * x
            - 60 * x3
            - 60 * x * (y2)
            + 210 * x5
            + 420 * (x3) * (y2)
            + 210 * x * (y4)
            - 280 * x7
            - 840 * (x5) * (y2)
            - 840 * (x3) * (y4)
            - 280 * x * (y6)
            + 126 * x9
            + 504 * (x7) * (y2)
            + 756 * (x5) * (y4)
            + 504 * (x3) * (y6)
            + 126 * x * (y8)
        )  # m = -1   n = 9
        Z51 = c[50] * (
            5 * y
            - 60 * y3
            - 60 * y * (x2)
            + 210 * y5
            + 420 * (y3) * (x2)
            + 210 * y * (x4)
            - 280 * y7
            - 840 * (y5) * (x2)
            - 840 * (y3) * (x4)
            - 280 * y * (x6)
            + 126 * y9
            + 504 * (y7) * (x2)
            + 756 * (y5) * (x4)
            + 504 * (y3) * (x6)
            + 126 * y * (x8)
        )  # m = -1   n = 9
        Z52 = c[51] * (
            -20 * y3
            + 60 * y * (x2)
            + 105 * y5
            - 210 * (y3) * (x2)
            - 315 * y * (x4)
            - 168 * y7
            + 168 * (y5) * (x2)
            + 840 * (y3) * (x4)
            + 504 * y * (x6)
            + 84 * y9
            - 504 * (y5) * (x4)
            - 672 * (y3) * (x6)
            - 252 * y * (x8)
        )  # m = 3  n = 9
        Z53 = c[52] * (
            21 * y5
            - 210 * (y3) * (x2)
            + 105 * y * (x4)
            - 56 * y7
            + 504 * (y5) * (x2)
            + 280 * (y3) * (x4)
            - 280 * y * (x6)
            + 36 * y9
            - 288 * (y7) * (x2)
            - 504 * (y5) * (x4)
            + 180 * y * (x8)
        )  # m = 5     n = 9
        Z54 = c[53] * (
            -8 * y7
            + 168 * (y5) * (x2)
            - 280 * (y3) * (x4)
            + 56 * y * (x6)
            + 9 * y9
            - 180 * (y7) * (x2)
            + 126 * (y5) * (x4)
            - 252 * (y3) * (x6)
            - 63 * y * (x8)
        )  # m = 7     n = 9
        Z55 = c[54] * (
            y9 - 36 * (y7) * (x2) + 126 * (y5) * (x4) - 84 * (y3) * (x6) + 9 * y * (x8)
        )  # m = 9       n = 9
        Z56 = c[55] * (
            10 * (x9) * y
            - 120 * (x7) * (y3)
            + 252 * (x5) * (y5)
            - 120 * (x3) * (y7)
            + 10 * x * (y9)
        )  # m = -10   n = 10
        Z57 = c[56] * (
            72 * (x7) * y
            - 504 * (x5) * (y3)
            + 504 * (x3) * (y5)
            - 72 * x * (y7)
            - 80 * (x9) * y
            + 480 * (x7) * (y3)
            - 480 * (x3) * (y7)
            + 80 * x * (y9)
        )  # m = -8    n = 10
        Z58 = c[57] * (
            270 * (x9) * y
            - 360 * (x7) * (y3)
            - 1260 * (x5) * (y5)
            - 360 * (x3) * (y7)
            + 270 * x * (y9)
            - 432 * (x7) * y
            + 1008 * (x5) * (y3)
            + 1008 * (x3) * (y5)
            - 432 * x * (y7)
            + 168 * (x5) * y
            - 560 * (x3) * (y3)
            + 168 * x * (y5)
        )  # m = -6   n = 10
        Z59 = c[58] * (
            140 * (x3) * y
            - 140 * x * (y3)
            - 672 * (x5) * y
            + 672 * x * (y5)
            + 1008 * (x7) * y
            + 1008 * (x5) * (y3)
            - 1008 * (x3) * (y5)
            - 1008 * x * (y7)
            - 480 * (x9) * y
            - 960 * (x7) * (y3)
            + 960 * (x3) * (y7)
            + 480 * x * (y9)
        )  # m = -4   n = 10
        Z60 = c[59] * (
            30 * x * y
            - 280 * (x3) * y
            - 280 * x * (y3)
            + 840 * (x5) * y
            + 1680 * (x3) * (y3)
            + 840 * x * (y5)
            - 1008 * (x7) * y
            - 3024 * (x5) * (y3)
            - 3024 * (x3) * (y5)
            - 1008 * x * (y7)
            + 420 * (x9) * y
            + 1680 * (x7) * (y3)
            + 2520 * (x5) * (y5)
            + 1680 * (x3) * (y7)
            + 420 * x * (y9)
        )  # m = -2   n = 10
        Z61 = c[60] * (
            -1
            + 30 * x2
            + 30 * y2
            - 210 * x4
            - 420 * (x2) * (y2)
            - 210 * y4
            + 560 * x6
            + 1680 * (x4) * (y2)
            + 1680 * (x2) * (y4)
            + 560 * y6
            - 630 * x8
            - 2520 * (x6) * (y2)
            - 3780 * (x4) * (y4)
            - 2520 * (x2) * (y6)
            - 630 * y8
            + 252 * x10
            + 1260 * (x8) * (y2)
            + 2520 * (x6) * (y4)
            + 2520 * (x4) * (y6)
            + 1260 * (x2) * (y8)
            + 252 * y10
        )  # m = 0    n = 10
        Z62 = c[61] * (
            -15 * x2
            + 15 * y2
            + 140 * x4
            - 140 * y4
            - 420 * x6
            - 420 * (x4) * (y2)
            + 420 * (x2) * (y4)
            + 420 * y6
            + 504 * x8
            + 1008 * (x6) * (y2)
            - 1008 * (x2) * (y6)
            - 504 * y8
            - 210 * x10
            - 630 * (x8) * (y2)
            - 420 * (x6) * (y4)
            + 420 * (x4) * (y6)
            + 630 * (x2) * (y8)
            + 210 * y10
        )  # m = 2  n = 10
        Z63 = c[62] * (
            -35 * x4
            + 210 * (x2) * (y2)
            - 35 * y4
            + 168 * x6
            - 840 * (x4) * (y2)
            - 840 * (x2) * (y4)
            + 168 * y6
            - 252 * x8
            + 1008 * (x6) * (y2)
            + 2520 * (x4) * (y4)
            + 1008 * (x2) * (y6)
            - 252 * (y8)
            + 120 * x10
            - 360 * (x8) * (y2)
            - 1680 * (x6) * (y4)
            - 1680 * (x4) * (y6)
            - 360 * (x2) * (y8)
            + 120 * y10
        )  # m = 4     n = 10
        Z64 = c[63] * (
            -28 * x6
            + 420 * (x4) * (y2)
            - 420 * (x2) * (y4)
            + 28 * y6
            + 72 * x8
            - 1008 * (x6) * (y2)
            + 1008 * (x2) * (y6)
            - 72 * y8
            - 45 * x10
            + 585 * (x8) * (y2)
            + 630 * (x6) * (y4)
            - 630 * (x4) * (y6)
            - 585 * (x2) * (y8)
            + 45 * y10
        )  # m = 6    n = 10
        Z65 = c[64] * (
            -9 * x8
            + 252 * (x6) * (y2)
            - 630 * (x4) * (y4)
            + 252 * (x2) * (y6)
            - 9 * y8
            + 10 * x10
            - 270 * (x8) * (y2)
            + 420 * (x6) * (y4)
            + 420 * (x4) * (y6)
            - 270 * (x2) * (y8)
            + 10 * y10
        )  # m = 8    n = 10
        Z66 = c[65] * (
            -1 * x10
            + 45 * (x8) * (y2)
            - 210 * (x6) * (y4)
            + 210 * (x4) * (y6)
            - 45 * (x2) * (y8)
            + y10
        )  # m = 10   n = 10

        ZW = (
            Z1
            + Z2
            + Z3
            + Z4
            + Z5
            + Z6
            + Z7
            + Z8
            + Z9
            + Z10
            + Z11
            + Z12
            + Z13
            + Z14
            + Z15
            + Z16
            + Z17
            + Z18
            + Z19
            + Z20
            + Z21
            + Z22
            + Z23
            + Z24
            + Z25
            + Z26
            + Z27
            + Z28
            + Z29
            + Z30
            + Z31
            + Z32
            + Z33
            + Z34
            + Z35
            + Z36
            + Z37
            + Z38
            + Z39
            + Z40
            + Z41
            + Z42
            + Z43
            + Z44
            + Z45
            + Z46
            + Z47
            + Z48
            + Z49
            + Z50
            + Z51
            + Z52
            + Z53
            + Z54
            + Z55
            + Z56
            + Z57
            + Z58
            + Z59
            + Z60
            + Z61
            + Z62
            + Z63
            + Z64
            + Z65
            + Z66
        )
        return ZW
    

