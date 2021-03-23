"""
Filters for data analysis.
"""
import numpy as np
import pyccl as ccl
import pylab as plt
from numpy import fft



def pca_filter(field, nmodes, return_filter=False):
    """
    Apply Principal Component Analysis (PCA) filter to a field. This subtracts 
    off functions in the frequency direction that correspond to the highest 
    SNR modes of the empirical frequency-frequency covariance.
    
    See Sect. 3.2 of Alonso et al. [arXiv:1409.8667] for details.
    N.B. Proper inverse-noise weighting is not currently used.
    
    Parameters
    ----------
    field : array_like
        3D array containing the field that the filter will be applied to. 
        NOTE: This assumes that the 3rd axis of the array is frequency.
    
    nmodes : int
        Number of eigenmodes to filter out (modes are ordered by SNR).
    
    return_filter : bool, optional
        Whether to also return the linear FG filter operator and coefficients. 
        Default: False.
    
    Returns
    -------
    cleaned_field : array_like
        Foreground-cleaned field.
    
    U_fg : array_like, optional
        Foreground operator, shape (Nfreq, Nmodes). Only returned if 
        `return_operator = True`.    
    
    fg_amps : array_like, optional
        Foreground mode amplitudes per pixel, shape (Nmodes, Npix). Only 
        returned if `return_operator = True`.
    """
    # Calculate freq-freq covariance matrix
    d = field.reshape((-1, field.shape[-1])).T # (Nfreqs, Nxpix * Nypix)
    
    # Calculate average spectrum (avg. over pixels, as a function of frequency)
    d_mean = np.mean(d, axis=-1)[:,np.newaxis]
    
    # Calculate freq-freq covariance matrix
    x = d - d_mean
    cov = np.cov(x) # (Nfreqs x Nfreqs)
    
    # Do eigendecomposition of covariance matrix
    eigvals, eigvecs = np.linalg.eig(cov)
    
    # Sort by eigenvalue
    idxs = np.argsort(eigvals)[::-1] # reverse order (biggest eigenvalue first)
    eigvals = eigvals[idxs]
    eigvecs = eigvecs[:,idxs]
    
    # Construct foreground filter operator by keeping only nmodes eigenmodes
    U_fg = eigvecs[:,:nmodes] # (Nfreqs, Nmodes)
    
    # Calculate foreground amplitudes for each line of sight
    fg_amps = np.dot(U_fg.T, x) # (Nmodes, Npix)
    
    # Construct FG field and subtract from input data
    fg_field = np.dot(U_fg, fg_amps) + d_mean # Operator times amplitudes + mean
    fg_field = fg_field.T.reshape(field.shape)
    cleaned_field = field - fg_field
    
    # Return filtered field (and optionally the filter operator + amplitudes)
    if return_filter:
        return cleaned_field, U_fg, fg_amps
    else:
        return cleaned_field
    
