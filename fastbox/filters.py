"""
Filters for data analysis, e.g. foreground filtering.
"""
import numpy as np
import pyccl as ccl
import pylab as plt
from numpy import fft
from scipy.optimize import curve_fit
from sklearn.decomposition import FastICA, NMF, KernelPCA


def mean_spectrum_filter(field):
    """
    Subtract the mean from each frequency slice.
    
    Parameters
    ----------
    field : array_like
        3D array containing the field that the filter will be applied to. 
        NOTE: This assumes that the 3rd axis of the array is frequency.
    
    Returns
    -------
    sub_field : array_like
        3D array containing the field with the mean at each frequency 
        subtracted.
    """
    # Calculate freq-freq covariance matrix
    d = field.reshape((-1, field.shape[-1])) # (Nxpix * Nypix, Nfreqs)
    
    # Calculate average spectrum (avg. over pixels, as a function of frequency)
    d_mean = np.mean(d, axis=0)[np.newaxis,:]
    x = d - d_mean # mean-subtracted data
    return x.reshape(field.shape)


def angular_bandpass_filter(field, kmin, kmax, d=1.):
    """
    Apply a top-hat bandpass filter to the field in each frequency channel 
    (i.e. each 2D slice) of a datacube.
    
    Parameters
    ----------
    field : array_like
        3D array containing the field that the filter will be applied to. 
        NOTE: This assumes that the 3rd axis of the array is frequency.
    
    kmin, kmax : array_like
        The Fourier wavenumbers defining the edges of the bandpass filter. The 
        units are defined by ``fft.fftfreq``. The bandpass filter is applied to 
        the magnitude of the 2D wavevector, k_perp = sqrt(k_x^2 + k_y^2).
    
    d : float, optional
        The pixel width parameter used by ``fft.fftfreq``. Default: 1.
    
    Returns
    -------
    filtered_field : array_like
        3D array containing the field with the angular bandpass filter applied.
    """
    # 2D FFT of field in transverse direction only
    field_k = np.fft.fftn(field, axes=[0,1])
    
    # Get frequencies
    kx = fft.fftfreq(field.shape[0], d=d)
    kx, ky = np.meshgrid(kx, kx)
    k = np.sqrt(kx**2. + ky**2.)
    
    # Filter frequencies that are out of range
    field_k[~np.logical_and(k >= kmin, k < kmax)] *= 0.
    return np.fft.ifftn(field_k, axes=[0,1])


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
    # Reshape field
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
        Number of eigenmodes to filter out.
    
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
    # Subtract mean vs. frequency
    x = mean_spectrum_filter(field).reshape((-1, field.shape[-1])).T
    
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


def kernel_pca_filter(field, nmodes, return_filter=False, **kwargs_pca):
    """
    Apply a Kernel Principal Component Analysis (KPCA) filter to a field. This 
    subtracts off functions in the frequency direction that correspond to the 
    highest SNR modes of the empirical frequency-frequency covariance, with 
    some non-linear weighting by a specified kernel.
    
    (WARNING: Can use a lot of memory)

    Uses `sklearn.decomposition.KernelPCA`. For more details, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html

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

    **kwargs_pca : dict, optional
        Keyword arguments for the `sklearn.decomposition.KernelPCA`

    Returns
    -------
    cleaned_field : array_like
        Foreground-cleaned field.

    transformer : sklearn.decomposition.KernelPCA instance, optional
        Contains the PCA filter. Only returned if `return_operator = True`. 
        To get the foreground model, you can do the following: 
            ```
            x = field - mean_field # shape (Npix, Nfreq)
            x_trans = transformer.fit_transform(x.T) # mode amplitudes per pixel
            x_fg = transformer.inverse_transform(x_trans).T # foreground model
            ```
    """
    # Subtract mean vs. frequency
    x = mean_spectrum_filter(field).reshape((-1, field.shape[-1])).T

    # Build PCA model and get amplitudes for each mode per pixel
    transformer = KernelPCA(n_components=nmodes, fit_inverse_transform=True, **kwargs_pca)
    x_trans = transformer.fit_transform(x.T)
    
    # Manually perform inverse transform, using the remaining eigenmode with 
    # the smallest eigenvalue
    X = transformer.alphas_[:,-1:] * np.sqrt(transformer.lambdas_[-1:]) # = x_trans
    K = transformer._get_kernel(X, transformer.X_transformed_fit_[:,-1:])
    n_samples = transformer.X_transformed_fit_.shape[0]
    K.flat[::n_samples + 1] += transformer.alpha
    x_clean = np.dot(K, transformer.dual_coef_).reshape(field.shape)
    
    # Return FG-subtracted data (and, optionally, the PCA filter instance)
    if return_filter:
        return x_clean, transformer
    else:
        return x_clean


def kernel_pca_filter_legacy(field, nmodes, return_filter=False, **kwargs_pca):
    """
    Apply a Kernel Principal Component Analysis (KPCA) filter to a field. This 
    subtracts off functions in the frequency direction that correspond to the 
    highest SNR modes of the empirical frequency-frequency covariance, with 
    some non-linear weighting by a specified kernel.
    
    NOTE: The sklearn KernelPCA function changed behaviour sometime after 
    v.0.22, so this function doesn't seem to work very well any more.
    
    (WARNING: Can use a lot of memory)

    Uses `sklearn.decomposition.KernelPCA`. For more details, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html

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

    **kwargs_pca : dict, optional
        Keyword arguments for the `sklearn.decomposition.KernelPCA`

    Returns
    -------
    cleaned_field : array_like
        Foreground-cleaned field.

    transformer : sklearn.decomposition.KernelPCA instance, optional
        Contains the PCA filter. Only returned if `return_operator = True`. 
        To get the foreground model, you can do the following: 
            ```
            x = field - mean_field # shape (Npix, Nfreq)
            x_trans = transformer.fit_transform(x.T) # mode amplitudes per pixel
            x_fg = transformer.inverse_transform(x_trans).T # foreground model
            ```
    """
    # Subtract mean vs. frequency
    x = mean_spectrum_filter(field).reshape((-1, field.shape[-1])).T

    # Build PCA model and get amplitudes for each mode per pixel
    transformer = KernelPCA(n_components=nmodes, fit_inverse_transform=True, **kwargs_pca)
    x_trans = transformer.fit_transform(x.T)

    # Construct foreground operator
    x_fg = transformer.inverse_transform(x_trans).T

    # Subtract foreground operator
    x_clean = (x - x_fg).T.reshape(field.shape)

    # Return FG-subtracted data (and, optionally, the PCA filter instance)
    if return_filter:
        return x_clean, transformer
    else:
        return x_clean


def nmf_filter(field, nmodes, return_filter=False, **kwargs_nmf):
    """
    Apply a Non-Negative Matrix Factorisation (NMF) filter to a field. This 
    finds two non-negative matrices whose product approximates the (strictly 
    non-negative) input signal. 

    Uses `sklearn.decomposition.NMF`. For more details, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

    Parameters
    ----------
    field : array_like
        3D array containing the field that the filter will be applied to. 
        NOTE: This assumes that the 3rd axis of the array is frequency.

    nmodes : int
        Number of eigenmodes to filter out.

    return_filter : bool, optional
        Whether to also return the linear FG filter operator and coefficients. 
        Default: False.

    **kwargs_nmf : dict, optional
        Keyword arguments for the `sklearn.decomposition.NMF`

    Returns
    -------
    cleaned_field : array_like
        Foreground-cleaned field.

    transformer : sklearn.decomposition.NMF instance, optional
        Contains the NMF filter. Only returned if `return_operator = True`. 
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
    x = d

    # Build NMF model and get amplitudes for each mode per pixel
    transformer = NMF(n_components=nmodes, **kwargs_nmf)
    x_trans = transformer.fit_transform(x.T)

    # Construct foreground operator
    x_fg = transformer.inverse_transform(x_trans).T

    # Subtract foreground operator
    x_clean = (x - x_fg).T.reshape(field.shape)

    # Return FG-subtracted data (and, optionally, the NMF filter instance)
    if return_filter:
        return x_clean, transformer
    else:
        return x_clean



def bandpower_pca_filter(field, nbands, modes):
    """
    Generate a series of bandpass-filtered datacubes and then PCA filter each 
    of them. The number of modes subtracted can be chosen separately for each 
    sub-band.
    
    N.B. The bandpass filters are contiguous top-hat filters of equal width in 
    Fourier space.
    
    Parameters
    ----------
    field : array_like
        3D array containing the field that the filter will be applied to. 
        NOTE: This assumes that the 3rd axis of the array is frequency.
        
    nbands : int
        How many sub-bands to divide the band into.
    
    nmodes : array_like or int
        Number of eigenmodes to filter out in each sub-band.
        This can be an array with a value per sub-band, 
        or a single integer for all bands.
    
    Returns
    -------
    cleaned_field : array_like
        Foreground-cleaned field.
    """
    # Expand modes array if needed
    if isinstance(modes, (int, np.integer)):
        modes = modes * np.ones(nbands, dtype=int)
        
    # Check for correct number of modes/bin edges
    assert nbands == len(modes), \
        "len(modes) must equal nbands"
    
    # Get min/max frequencies and use to define the sub-bands
    kx = fft.fftfreq(field.shape[0], d=1.)
    kx, ky = np.meshgrid(kx, kx)
    k = np.sqrt(kx**2. + ky**2.)
    band_edges = np.linspace(np.min(k), np.max(k), nbands+1)
    
    # Mean-subtracted field
    x = mean_spectrum_filter(field)
    
    # Loop over bands
    bpf_cleaned = 0
    for i in range(len(band_edges)-1):
        # Apply bandpass filter
        bpf_cube = angular_bandpass_filter(x, 
                                           kmin=band_edges[i], 
                                           kmax=band_edges[i+1])
        
        # Apply PCA cleaning
        _bpf_cleaned = fastbox.filters.pca_filter(bpf_cube, 
                                                  nmodes=modes[i], 
                                                  return_filter=False)
        bpf_cleaned += _bpf_cleaned
    return bpf_cleaned


