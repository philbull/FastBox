"""
Filters for data analysis, e.g. foreground filtering.
"""
import numpy as np
import pyccl as ccl
import pylab as plt
from numpy import fft
from scipy.optimize import curve_fit
from sklearn.decomposition import FastICA


def pca_filter(field, nmodes, fit_powerlaw=False, return_filter=False):
    """
    Apply a Principal Component Analysis (PCA) filter to a field. This 
    subtracts off functions in the frequency direction that correspond to the 
    highest SNR modes of the empirical frequency-frequency covariance.
    
    Note that the mean as a function of frequency (i.e. average over x and y 
    pixels for each frequency channel) is subtracted from the data before 
    calculating the PCA modes. The mean is then added back into the foreground 
    model at the end.
    
    See Sect. 3.2 of Alonso et al. [arXiv:1409.8667] for details.
    N.B. Proper inverse-noise weighting is not currently used.
    
    Parameters
    ----------
    field : array_like
        3D array containing the field that the filter will be applied to. 
        NOTE: This assumes that the 3rd axis of the array is frequency.
    
    nmodes : int
        Number of eigenmodes to filter out (modes are ordered by SNR).
    
    fit_powerlaw : bool, optional
        If True, fit a power-law to the mean as a function of frequency. This 
        may help prevent over-fitting of the mean relation. If False, the 
        simple mean as a function of frequency will be used. Default: False.
    
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
    
    # Fit power law to the mean and subtract that instead
    # (FIXME: Needs to be tested more thoroughly)
    if fit_powerlaw:
        freqs = np.linspace(1., 10., d.shape[0])
        
        def fn(nu, amp, beta):
            return amp * (nu / nu[0])**beta
        
        p0 = [d_mean[0][0], -2.7]
        pfit, _ = curve_fit(fn, freqs, d_mean.flatten(), p0=p0)
        d_mean = fn(freqs, pfit[0], pfit[1])[:,np.newaxis] # use power-law fit instead
        
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



def ica_filter(field, nmodes, return_filter=False, **kwargs_ica):
    """
    Apply an Independent Component Analysis (ICA) filter to a field. This 
    subtracts off functions in the frequency direction that correspond to the 
    highest SNR *statistically independent* modes of the empirical 
    frequency-frequency covariance.
    
    Uses `sklearn.decomposition.FastICA`. For more details, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
    
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
    
    **kwargs_ica : dict, optional
        Keyword arguments for the `sklearn.decomposition.FastICA`
    
    Returns
    -------
    cleaned_field : array_like
        Foreground-cleaned field.
    
    transformer : sklearn.decomposition.FastICA instance, optional
        Contains the ICA filter. Only returned if `return_operator = True`. 
        To get the foreground model, you can do the following: 
            ```
            x = field - mean_field # shape (Npix, Nfreq)
            x_trans = transformer.fit_transform(x.T) # mode amplitudes per pixel
            x_fg = transformer.inverse_transform(x_trans).T # foreground model
            ```
    """
    # Calculate freq-freq covariance matrix
    d = field.reshape((-1, field.shape[-1])).T # (Nfreqs, Nxpix * Nypix)
    
    # Calculate average spectrum (avg. over pixels, as a function of frequency)
    d_mean = np.mean(d, axis=-1)[:,np.newaxis]
    x = d - d_mean # mean-subtracted data
    
    # Build ICA model and get amplitudes for each mode per pixel
    transformer = FastICA(n_components=nmodes, **kwargs_ica)
    x_trans = transformer.fit_transform(x.T)
    
    # Construct foreground operator
    x_fg = transformer.inverse_transform(x_trans).T
    
    # Subtract foreground operator
    x_clean = (x - x_fg).T.reshape(field.shape)
    
    # Return FG-subtracted data (and, optionally, the ICA filter instance)
    if return_filter:
        return x_clean, transformer
    else:
        return x_clean


