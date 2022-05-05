
import numpy as np
from scipy.linalg import sqrtm
from scipy.sparse.linalg import cg as conjgrad
from scipy.optimize import minimize, Bounds


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
        + ridge_var * np.eye(freqs.size) # ridge adjustment
    return cov


def gaussian_cr_1d(d, w, S, N, realisations=1, add_noise=True, 
                   precondition=True, cg_maxiter=1e4, verbose=True):
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
        
        cg_maxiter (int, optional):
            Max. number of conjugate gradient iterations for the solver.
        
        verbose (bool, optional):
            Whether to print progress messages.
        
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
    sqrtS = sqrtm(S)
    sqrtN = sqrtm(N)
    Sinv = np.linalg.inv(S)
    Ninv = np.linalg.inv(N)
    sqrtSinv = sqrtm(Sinv)
    
    # Empty solution array
    solns = np.zeros((realisations, Npix, Nfreq), dtype=np.complex128)
        
    # Loop over pixels
    for j in range(Npix):
        if verbose:
            print("    Pixel %d / %d" % (j+1, Npix))
        
        # Flagged inverse noise matrix
        Ninvw = w[j,:].T * Ninv * w[j,:] # noise covariance with flags applied
        sqrtNinvw = sqrtm(Ninvw)
        
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
            omegaN = np.random.randn(Nfreq)
            omegaS = np.random.randn(Nfreq)
            
            # Add random part to RHS and solve using CG
            b_cr = b + omegaN + sqrtS @ sqrtNinvw @ omegaS
            x, info2 = conjgrad(A, b_cr, maxiter=cg_maxiter, M=Ainv)
            #solns[i,j,:] = x
            
            # Add noise realisation if requested; otherwise just rescale x 
            # variable by factor of sqrtS to get realisation 's'
            if add_noise:
                solns[i,j,:] = sqrtS @ x + (sqrtN @ omegaN).T[0]
            else:
                solns[i,j,:] = sqrtS @ x
            
    return solns


def trim_flagged_channels(w, x):
    """
    Remove flagged channels from a 1D or 2D (square) array. This is 
    a necessary pre-processing step for LSSA.

    Parameters:
        w (array_like):
            1D array of mask values, where 1 means unmasked and 0 means 
            masked.
        
        x (array_like):
            1D or square 2D array to remove the masked channels from.

    Returns:
        xtilde (array_like):
            Input array with the flagged channels removed.
    """
    # Check inputs
    assert np.shape(x) == (w.size,) or np.shape(x) == (w.size, w.size), \
        "Input array must have shape (w.size) or (w.size, w.size)"

    # 1D case
    if len(x.shape) == 1:
        return x[w == 1.]
    else:
        return x[:,w == 1.][w == 1.,:]


def _model_ap(amp, phase, tau, freqs):
    return amp * np.exp(2.*np.pi*1.j*tau*freqs + 1.j*phase)

def _model_aa(A_re, A_im, tau, freqs):
    return (A_re + 1.j*A_im) * np.exp(2.*np.pi*1.j*tau*freqs)

def lssa_fit_modes(d, freqs, invcov=None, fit_amp_phase=True, tau=None, 
                   minimize_method='L-BFGS-B', taper=False):
    r"""
    Perform a weighted LSSA fit to masked complex 1D data.

    NOTE: The input data/covariance should have already had the flagged 
    channels removed. Use the `trim_flagged_channels()` function to do 
    this.
    
    The log-likelihood for each sinusoid takes the assumed form:
    
    $\log L_n = \tilde{x}^\dagger \tilde{C}^{-1} \tilde{x}$
    
    where $\tau_n = n / \Delta \nu$, $\Delta \nu$ is the bandwidth, and 
    
    $x = [d - A \exp(2 \pi i \nu \tau_n + i\phi)]$.

    The tilde denotes vectors/matrices from which the masked channels 
    (rows/columns) have been removed entirely.
    
    Parameters:
        d (array_like):
            Complex data array that has already had flagged channels removed.
        
        freqs (array_like):
            Array of frequency values, in MHz. Used to get tau values in 
            the right units only. Flagged channels must have already been 
            removed.
        
        invcov (array_like):
            Inverse of the covariance matrix (flagged channels must have been 
            removed before inverting).

        fit_amp_phase (bool, optional):
            If True, fits the (real) amplitude and (real) phase parameters 
            for each sinusoid. If False, fits the real and imaginary amplitudes.
        
        tau (array_like, optional):
            Array of tau modes to fit. If `None`, will use `fftfreq()` to 
            calculate the tau values. Units: nanosec.
        
        taper (array_like, optional):
            If specified, multiplies the data and sinusoid model by a taper 
            function to enforce periodicity. The taper should be evaluated 
            at the locations specified in `freqs`
        
        minimize_method (str, optional):
            Which SciPy minimisation method to use. Default: `'L-BFGS-B'`.
    
    Returns:
        tau (array_like):
            Wavenumbers, calculated as tau_n = n / L, in nanoseconds.
            
        param1, param2 (array_like):
            If `fit_amp_phase` is True, these are the best-fit amplitude and 
            phase of the sinusoids. Otherwise, they are the real and imaginary 
            amplitudes of the sinusoids.
    """
    # Get shape of data etc.
    bandwidth = (freqs[-1] - freqs[0]) / 1e3 # assumed MHz, convert to GHz
    assert d.size == invcov.shape[0] == invcov.shape[1] == freqs.size, \
        "Data, inv. covariance, and freqs array must have same number of channels"
    
    # Calculate tau values
    if tau is None:
        tau = np.fft.fftfreq(n=freqs.size, d=freqs[1]-freqs[0]) * 1e3 # nanosec
    
    # Taper
    if taper is None:
        taper = 1.
    else:
        assert taper.size == freqs.size, \
            "'taper' must be evaluated at locations given in 'freqs'"
    
    # Log-likelihood (or log-posterior) function
    def loglike(p, n):
        if fit_amp_phase:
            m = _model_ap(amp=p[0], phase=p[1], tau=tau[n], freqs=freqs)
        else:
            m = _model_aa(A_re=p[0], A_im=p[1], tau=tau[n], freqs=freqs)
        
        # Calculate residual and log-likelihood
        x = taper * (d - m)
        logl = 0.5 * np.dot(x.conj(), np.dot(invcov, x))
        return logl.real # Result should be real
    
    # Set appropriate bounds for fits
    max_abs = np.max(np.abs(d))
    if fit_amp_phase:
        bounds = [(-100.*max_abs, 100.*max_abs), (0., 2.*np.pi)]
    else:
        bounds = [(-100.*max_abs, 100.*max_abs), (-100.*max_abs, 100.*max_abs)]
    
    # Do least-squares fit for each tau
    param1 = np.zeros(tau.size)
    param2 = np.zeros(tau.size)
    
    for n in range(tau.size):
        p0 = np.zeros(2)

        # Rough initial guess
        if fit_amp_phase:
            p0[0] = 0.2 * np.max(np.abs(d))
            p0[1] = 0.5 * np.pi
        else:
            p0[0] = 0.2 * np.max(d.real) # rough guess at amplitude
            p0[1] = 0.2 * np.max(d.imag)
        
        # Least-squares fit for mode n
        result = minimize(loglike, p0, args=(n,), 
                          method=minimize_method, 
                          bounds=bounds)
        param1[n], param2[n] = result.x
    
    return tau, param1, param2


def lssa_decorr_matrix(w, tau, freqs):
    """
    Calculate rotation matrix from Eq. 8 of Bryna Hazelton's LSSA note, 
    needed to decorrelate the real and imaginary amplitudes of the LSSA 
    cosine/sine modes.
    
    To use this matrix to decorrelate the amplitudes, do:
    `np.dot(rot, [A_real, A_imag])`

    Note that you can use the `w` and `freqs` arrays with or without the 
    flagged channels removed by `trim_flagged_channels()`; the results 
    should be equivalent.
    
    Parameters:
        w (array_like):
            Mask vector, 1 for unmasked, 0 for masked.
        
        tau (float):
            Delay wavenumber, in nanoseconds.
        
        freqs (array_like):
            Frequency array, in MHz.
    
    Returns:
        rot (array_like):
            Rotation matrix to be applied to the amplitude vector.
            
        eigvals (array_like):
            Eigenvalues of mode correlation matrix. Multiply the 
            variance of the mode, sigma^2, with these eigenvalues 
            to get the new variances (sigma1^2, sigma2^2); see 
            Eq. 9 of Bryna's note.
    """
    # Sine and cosine terms with mask (factor of 1e3 converts MHz->GHz)
    cos = w*np.cos(2.*np.pi*tau*freqs/1e3)
    sin = w*np.sin(2.*np.pi*tau*freqs/1e3)
    
    # Covariance (overlap) matrix
    cov = np.zeros((2, 2))
    cov[0,0] = np.sum(cos*cos)
    cov[0,1] = cov[1,0] = np.sum(cos*sin)
    cov[1,1] = np.sum(sin*sin)
    
    # Calculate rotation angle directly
    theta = 0.5 * np.arctan2(2.*np.sum(cos*sin), 
                             np.sum(cos*cos) - np.sum(sin*sin))
    rot = np.array([[np.cos(theta), np.sin(theta)], 
                     [-np.sin(theta), np.cos(theta)]])
    rinv = np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]])
    eigvals = np.diag(np.dot(rot, np.dot(cov, rinv)))

    return rot, eigvals


def lssa_pspec(A_re, A_im, w, tau, freqs, decorrelate_amps=True):
    """
    Calculate the LSSA power spectrum, by using Bryna's decorrelation 
    scheme to re-weight the real and imaginary amplitudes.

    Parameters:
        A_re, A_im (array_like):
            Real and imaginary amplitudes of the LSSA modes, from 
            `lssa_fit_modes()`.

        w (array_like):
            Mask array, with 1 for unmasked and 0 for masked.

        tau (array_like):
            Wavenumbers of the LSSA modes, in nanoseconds.

        freqs (array_like):
            Frequencies of the 


    """
    ps = np.zeros(tau.size)
    
    # Loop over tau modes
    for i, t in enumerate(tau):
        # Get decorrelation matrix and eigenvalues
        rot, eigvals = lssa_decorr_matrix(w=w, tau=t, freqs=freqs)
        
        # Apply decorrelation rotation
        A1, A2 = np.matmul(rot, np.array([A_re[i], A_im[i]]))
        
        # Construct power spectrum (c.f. Eq. 12 of Bryna's note)
        # Multiplied num. and denom. by each eigval squared to avoid 1/0
        ps[i] = ((A1 * eigvals[1])**2. + (A2 * eigvals[0])**2.) \
              / (eigvals[0]**2. + eigvals[1]**2.)
    return ps