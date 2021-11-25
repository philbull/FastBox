
import numpy as np
from scipy.sparse.linalg import cg as conjgrad


def simple_signal_cov(freqs, amplitude, width, ridge_var=1e-10):
    """
    Simple signal covariance model, using a Gaussian correlation function with 
    some width (correlation length).
    
    NOTE: A small ridge adjustment is added (1e-10 * I).
    
    Parameters:
        freqs (array_like):
            1D array of frequencies, in frequency units.
        amplitude (float):
            Amplitude of the covariance, in the units of the data (squared).
        width (float):
            Correlation length of the Gaussian, in frequency units.
        ridge_var (float):
            Ridge adjustment (variance to add along the diagonal).
    
    Returns:
        cov (array_like):
            Signal covariance model.
    """
    nu, nup = np.meshgrid(freqs, freqs)
    cov = amplitude * np.exp(-0.5 * (nu - nup)**2. / width**2.) \
        + ridge_var * np.eye(s) # ridge adjustment
    return cov


def gaussian_cr_1d(d, w, S, N, realisations=1, add_noise=True, precondition=True):
    """
    Returns Gaussian constrained realizations for a flagged 1D data vector with 
    signal prior covariance S and noise covariance N.
    
    NOTE: If `noisy=True`, a noise realisation is added to all elements of the 
    constrained realisation, meaning that the realisation will not match the 
    data in the unflagged region.
    
    If the desire is to in-paint flagged regions, you will need to select only 
    this region of the output vector.
    
    NOTE: A conjugate gradient solver is used to solve the constrained 
    realisation equation.
    
    NOTE: The constrained realisation equation that is being solved is a 
    rescaled version, where the signal realisation s = S^1/2 x, and the 
    equation being solved to find x has the form Ax = b, where
    
        A = S^1/2 (w^T N^-1 w)^1/2 S^1/2 + I
        b = S^1/2 N^-1 (w d)
          + omegaN + S^1/2 (w^T N^-1 w)^1/2 omegaS
          
    where omegaN and omegaS are 1D vectors of unit Gaussian random draws.
    
    Parameters:
        d (array_like):
            Data vector (assumed real), of shape (Npix, Nfreqs). The GCR solver 
            will treat each pixel independently.
        
        w (array_like):
            Flagging/mask vector (1 for unflagged data, 0 for flagged data), 
            with shape (Npix, Nfreq).
        
        S (array_like):
            Signal prior covariance matrix. Real-valued, with shape 
            (Nfreq, Nfreq).
        
        N (array_like):
            Noise covariance matrix. Real-valued, with shape (Nfreq, Nfreq).
        
        realisations (int, optional):
            Number of random realisations to return.
        
        add_noise (bool, optional):
            Whether to add a noise realisation into the returned constrained 
            realisations.
        
        precondition (bool, optional):
            If True, use the pseudo-inverse of the CG operator matrix as a 
            preconditioner in the conjugate gradient solver. This is calculated 
            only once at the start of the function, but may be slow if many 
            frequency channels are used.
    
    Returns:
        solutions (array_like):
            Array of solution, of shape (realisations, Npix, Nfreq).
    """
    # Check shape of inputs
    assert len(d.shape) == len(w.shape) == 2, \
        "d and w must have shape (Npix, Nfreq)"
    Npix, Nfreq = d.shape
    assert S.shape == (Nfreq, Nfreq), "S must have shape (Nfreq, Nfreq)"
    assert N.shape == (Nfreq, Nfreq), "N must have shape (Nfreq, Nfreq)"
    assert np.all(np.isreal(S)), "S must be real"
    assert np.all(np.isreal(N)), "N must be real"
    
    # Construct matrix operators ahead of time
    sqrtS = sp.linalg.sqrtm(S)
    sqrtN = sp.linalg.sqrtm(N)
    Sinv = np.linalg.inv(S)
    Ninv = np.linalg.inv(N)
    sqrtSinv = sp.linalg.sqrtm(Sinv)
    
    # Empty solution array
    solns = np.zeros((realisations, Npix, Nfreq), dtype=np.complex)
        
    # Loop over pixels
    for j in range(Npix):
        
        # Flagged inverse noise matrix
        Ninvw = w[j,:].T * Ninv * w[j,:] # noise covariance with flags applied
        sqrtNinvw = sp.linalg.sqrtm(Ninvw)
        
        # Explicitly construct LHS operator matrix and RHS data-dep. part
        A = sqrtS @ Ninvw @ sqrtS + np.eye(Nfreq)
        if precondition:
            Ainv = np.linalg.pinv(A) # use pseudo-inverse as preconditioner
        else:
            Ainv = None
        b = sqrtS @ Ninv @ (w[j,:] * d[j,:]).T
        
        # Loop over realisations
        for i in range(realisations):    
            
            # Unit Gaussian draws for random realisations
            omegaN = np.random.randn(Nfreq,1)
            omegaS = np.random.randn(Nfreq,1)
            
            # Add random part to RHS and solve using CG
            b_cr = b + omegaN + sqrtS @ sqrtNinvw @ omegaS
            x, info2 = conjgrad(A, b_cr, maxiter=1e5, M=Ainv)
            #solns[i,j,:] = x
            
            # Add noise realisation if requested; otherwise just rescale x 
            # variable by factor of sqrtS to get realisation 's'
            if add_noise:
                solns[i,j,:] = sqrtS @ x + (sqrtN @ omegaN).T[0]
            else:
                solns[i,j,:] = sqrtS @ x
            
    return solns
