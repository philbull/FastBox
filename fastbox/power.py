import numpy as np
import numpy.fft as fft
import pyccl as ccl

class Power(object):
    def __init__(self, box):
        """
        An object to manage power spectrum estimation.
        
        Parameters:
            box (CosmoBox):
                Object containing a simulation box.
        """
        self.box = box

    def _isotropic_k_range(self):
        """Return fundamental and conservative isotropic Nyquist wavenumbers.

        A spherically averaged spectrum is only alias-safe below the smallest
        axis-specific Nyquist frequency. This is essential for anisotropic
        boxes, where using ``Nz / min(L)`` can incorrectly include modes that
        are unavailable along a transverse direction.
        """
        k_fundamental = 2.0 * np.pi / max(self.box.Lx, self.box.Ly, self.box.Lz)
        k_nyquist = min(
            np.pi * self.box.Nx / self.box.Lx,
            np.pi * self.box.Ny / self.box.Ly,
            np.pi * self.box.Nz / self.box.Lz,
        )
        return k_fundamental, k_nyquist

    def get_window_correction_grid(self, method=None):
        """
        Calculates the 3D squared window function |W(k)|^2 for mass assignment correction.

        It implements equation 18 of https://arxiv.org/abs/astro-ph/0409240
        
        Parameters: 
        -----------
        method : 'CIC' of 'NGP' 
            Method to consider
            
        Returns:
        --------
        W2_k : ndarray
            Window function.
        
        """
        if method is None or method.upper() not in ['NGP', 'CIC']:
            raise ValueError("Valid methods: 'NGP' and 'CIC'")
            
        p = 1 if method.upper() == 'NGP' else 2
        
        # 1. Get the 1D sinc function along one axis
        w1d_x = np.sinc(np.fft.fftfreq(self.box.Nx))
        w1d_y = np.sinc(np.fft.fftfreq(self.box.Ny))
        w1d_z = np.sinc(np.fft.fftfreq(self.box.Nz))
        
        # 2. Square the 1D window
        w1d_sq_x = w1d_x**2
        w1d_sq_y = w1d_y**2
        w1d_sq_z = w1d_z**2
        
        # 3. Broadcast to 3D axes (X, Y, Z)
        Wx2 = w1d_sq_x[:, None, None]
        Wy2 = w1d_sq_y[None, :, None]
        Wz2 = w1d_sq_z[None, None, :]
        
        # 4. Combine and raise to the assignment scheme power
        W2_k = (Wx2 * Wy2 * Wz2)**p
        
        return W2_k

    def unweighted_power(self, delta_x1, delta_x2=None):
        """
        Calculates the 1D spherically averaged auto or cross power spectrum.
        
        Parameters: 
        -----------
        delta_x1 : ndarray 
            overdensity mesh
        delta_x2 : ndarray
            Second overdensity mesh. Default None. If not None, returns cross-power
        
        Returns: 
        --------
        kc     : ndarray
            Center of k bins
        vals   : ndarray
            power spectrum (auto or cross, depending on delta_x2)
        stddev :
            simple calculation of standard deviation of the estimator
        """
        return self.weighted_power(delta_x1, delta_x2, w1=None, w2=None)

    def weighted_power(self, delta_x1, delta_x2=None, w1=None, w2=None):
        """
        Calculates the 1D spherically averaged auto or cross power spectrum, 
        with optional weighting.
        
        Parameters: 
        -----------
        delta_x1 : ndarray 
            overdensity mesh
        delta_x2 : ndarray
            Second overdensity mesh. Default None. If not None, returns cross-power
        w1, w2 : ndarray, optional
            Weights for each mesh.
        
        Returns: 
        --------
        kc     : ndarray
            Center of k bins
        vals   : ndarray
            power spectrum (auto or cross, depending on delta_x2)
        stddev :
            simple calculation of standard deviation of the estimator
        """
        if delta_x1 is None:
            raise ValueError('Need to specify field delta_x1')

        # Apply weights
        d1 = delta_x1.copy()
        if w1 is not None:
            d1 *= w1
        
        # FFT of the first field
        delta_k1 = fft.fftn(d1)

        # ---------------------------------------------------------
        # AUTO-CORRELATION
        # ---------------------------------------------------------
        if delta_x2 is None:
            print('Calculating auto-correlation power spectrum...')
            pk = delta_k1 * np.conj(delta_k1)
            pk = pk.real / self.box.boxfactor
            
            # Correction for weights (approximate)
            if w1 is not None:
                pk /= np.mean(w1**2)

        # ---------------------------------------------------------
        # CROSS-CORRELATION
        # ---------------------------------------------------------
        else:
            print('Calculating cross-correlation power spectrum')
            d2 = delta_x2.copy()
            if w2 is not None:
                d2 *= w2
            
            delta_k2 = fft.fftn(d2)
            
            # Cross power is the real part of (delta_1 * conj(delta_2))
            pk = delta_k1 * np.conj(delta_k2)
            pk = pk.real / self.box.boxfactor
            
            # Correction for weights (approximate)
            if w1 is not None or w2 is not None:
                W1 = w1 if w1 is not None else 1.0
                W2 = w2 if w2 is not None else 1.0
                pk /= np.mean(W1 * W2)
            

        # ---------------------------------------------------------
        # K-BINNING
        # ---------------------------------------------------------
        k_min, k_nyq = self._isotropic_k_range()
        k_bin = k_min
        
        bins = np.arange(k_min, k_nyq + k_bin, k_bin)
        kc = 0.5 * (bins[1:] + bins[:-1])
        
        vals = np.zeros(kc.size)
        stddev = np.zeros(kc.size)
        
        idxs = np.digitize(self.box.k.flatten(), bins)
        pk_flat = pk.flatten()
        
        for i in range(1, bins.size):
            ii = (idxs == i)
            if np.any(ii):
                vals[i-1] = np.mean(pk_flat[ii])
                stddev[i-1] = np.std(pk_flat[ii]) / np.sqrt(np.sum(ii))
            else:
                vals[i-1] = np.nan
                stddev[i-1] = np.nan
        print('DONE')
        
        return kc, vals, stddev

    def model_obs_power_IM(self, km, pkm, bias, Tb, sigdeg, rsd=True, sigma_nl=120,fog_model='lorentzian'):
        """
        Computes the theoretical observational power spectrum by forward-modeling 
        the theoretical matter power spectrum on the 3D FFT grid.
        Includes Beam smoothing, Channel smoothing, FFT Discretization, and RSD.
        
        Parameters:
        -----------
        km, pkm : array_like
            The 1D theoretical matter power spectrum (wavenumbers and power).
        bias : float
            The linear bias factor (b)
        Tb : float
            The mean brightness temperature (signal amplitude)
        sigdeg : float
            Standard deviation of the Gaussian beam in degrees. Set 0 to ignore beam.
        rsd : Bool
            Set True to apply Kaiser and FoG effects.
        sigma_nl : float
            Velocity dispersion in km/s for Finger of God damping. Set 0 to ignore FoG.
            
        Returns:
        --------
        kc, pk_model, stddev : ndarray
            The binned, forward-modeled 1D power spectrum.
        """
        
        # =======================================================
        # 1. Calculate Physical Scales & Cosmology (R_beam, R_chan, R_nl)
        # =======================================================
        z = self.box.redshift
        scale_factor = 1.0 / (1.0 + z)
        h = self.box.cosmo['h']
        Hz = 100. * h * ccl.h_over_h0(self.box.cosmo, scale_factor)
        # Transverse Beam Scale
        if sigdeg > 0:
            chi_mpc = ccl.comoving_radial_distance(self.box.cosmo, scale_factor)
            R_beam = np.radians(sigdeg) * (chi_mpc )
        else:
            R_beam = 0.0
        '''    
        # LoS Channel Scale
        if channel:
            freqs = self.box.freq_array()
            dnu_mhz = np.abs(np.mean(np.diff(freqs)))
            nu_21 = 1420.40575177
            c_kms = 299792.458
            R_chan = (c_kms * (1.0 + z)**2) / (Hz * nu_21) * dnu_mhz * h
        else:
            R_chan = 0.0
        '''
        # RSD Linear Growth Rate (f) and FoG Scale (R_nl)
        if rsd:
            f = ccl.growth_rate(self.box.cosmo, scale_factor)
            if sigma_nl > 0:
                # Convert km/s to comoving Mpc/h
                R_nl = (sigma_nl * (1.0 + z) / Hz)
            else:
                R_nl = 0.0
        
        # =======================================================
        # 2. Create the exact 3D Fourier Grid
        # =======================================================
        kx = 2.0 * np.pi * np.fft.fftfreq(self.box.Nx, d=self.box.Lx / self.box.Nx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.box.Ny, d=self.box.Ly / self.box.Ny)
        kz = 2.0 * np.pi * np.fft.fftfreq(self.box.Nz, d=self.box.Lz / self.box.Nz)
        
        kx3d = kx[:, None, None]
        ky3d = ky[None, :, None]
        kz3d = kz[None, None, :]
        
        # k_perp (X, Y) and k_parallel (Z)
        k_perp_sq = kx3d**2 + ky3d**2
        k_par_sq = kz3d**2
        k_mag_sq = k_perp_sq + k_par_sq
        k_mag = np.sqrt(k_mag_sq)
        
        # =======================================================
        # 3. Evaluate 3D Theory & Apply RSD + Damping Factors
        # =======================================================
        # Interpolate the theoretical 1D P(k) onto the 3D grid
        pk_3d = np.interp(k_mag.flatten(), km, pkm).reshape(k_mag.shape)
        
        # --- RSD (Kaiser + FoG) ---
        if rsd:
            # Safely calculate mu^2 = k_parallel^2 / k_mag^2 (avoid division by zero at k=0)
            # FIXED: Used np.zeros_like(k_mag_sq) to match the broadcast shape
            mu_sq = np.divide(k_par_sq, k_mag_sq, out=np.zeros_like(k_mag_sq), where=(k_mag_sq != 0))
            
            # Kaiser Factor
            rsd_factor = (bias + f * mu_sq)**2
            
            # FoG Damping (velocity dispersion)
            if sigma_nl > 0:
                if fog_model == 'lorentzian':
                    D2_fog = 1.0 / (1.0 + k_par_sq * R_nl**2)
                elif fog_model == 'gaussian':
                    D2_fog = np.exp(-k_par_sq * R_nl**2)

            else:
                D2_fog = 1.0
                
            pk_3d = pk_3d * rsd_factor * D2_fog
        else:
            # No RSD: just use isotropic bias
            pk_3d = pk_3d * (bias**2)


        pk_3d = pk_3d * (Tb**2)
        
        # --- Observational Damping ---
        if sigdeg > 0:
            D2_beam = np.exp(-k_perp_sq * (R_beam**2))
        else:
            D2_beam = 1.0
        '''    
        if channel:
            k_par = np.sqrt(k_par_sq)
            D2_chan = (np.sinc((k_par * R_chan) / (2.0 * np.pi)))**2
        else:
            D2_chan = 1.0
        '''    
        # Combine all effects
        pk_3d_obs = pk_3d * D2_beam# * D2_chan 
        
        # =======================================================
        # 4. Bin the 3D Model into 1D k-shells
        # =======================================================
        k_min, k_nyq = self._isotropic_k_range()
        k_bin = k_min
        
        bins = np.arange(k_min, k_nyq + k_bin, k_bin)
        kc = 0.5 * (bins[1:] + bins[:-1])
        
        vals = np.zeros(kc.size)
        stddev = np.zeros(kc.size)
        
        idxs = np.digitize(k_mag.flatten(), bins)
        pk_flat = pk_3d_obs.flatten()
        
        for i in range(1, bins.size):
            ii = (idxs == i)
            if np.any(ii):
                vals[i-1] = np.mean(pk_flat[ii])
                stddev[i-1] = np.std(pk_flat[ii]) / np.sqrt(np.sum(ii))
            else:
                vals[i-1] = np.nan
                stddev[i-1] = np.nan
        print('DONE')
        
        return kc, vals, stddev

    def model_obs_power_gal(self, km, pkm, bias, MAS='NGP', rsd=True, sigma_nl=120):
        """
        Computes the theoretical observational power spectrum for galaxies
        by forward-modeling the matter power spectrum on the 3D FFT grid.
        Includes Mass Assignment Scheme (MAS) window and RSD.
        
        Parameters:
        -----------
        km, pkm : array_like
            The 1D theoretical matter power spectrum (wavenumbers and power).
        bias : float
            The linear bias factor (b)
        MAS : 'NGP' or 'CIC' or None
            mass assignment method to correct.
        rsd : Bool
            Set True to apply Kaiser and FoG effects.
        sigma_nl : float
            Velocity dispersion in km/s for Finger of God damping. Set 0 to ignore FoG.
            
        Returns:
        --------
        kc, pk_model, stddev : ndarray
            The binned, forward-modeled 1D power spectrum.
        """
        
        # =======================================================
        # 1. Create the exact 3D Fourier Grid
        # =======================================================
        kx = 2.0 * np.pi * np.fft.fftfreq(self.box.Nx, d=self.box.Lx / self.box.Nx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.box.Ny, d=self.box.Ly / self.box.Ny)
        kz = 2.0 * np.pi * np.fft.fftfreq(self.box.Nz, d=self.box.Lz / self.box.Nz)
        
        kx3d = kx[:, None, None]
        ky3d = ky[None, :, None]
        kz3d = kz[None, None, :]
        
        # k_perp (X, Y) and k_parallel (Z)
        k_perp_sq = kx3d**2 + ky3d**2
        k_par_sq = kz3d**2
        k_mag_sq = k_perp_sq + k_par_sq
        k_mag = np.sqrt(k_mag_sq)

        # =======================================================
        # 2. Evaluate 3D Theory & Apply RSD 
        # =======================================================
        # Interpolate the theoretical 1D P(k) onto the 3D grid
        pk_3d = np.interp(k_mag.flatten(), km, pkm).reshape(k_mag.shape)
        
        # Cosmology for RSD
        z = self.box.redshift
        scale_factor = 1.0 / (1.0 + z)

        if rsd:
            h = self.box.cosmo['h']
            Hz = 100. * h * ccl.h_over_h0(self.box.cosmo, scale_factor)
            f = ccl.growth_rate(self.box.cosmo, scale_factor)
            
            if sigma_nl > 0:
                R_nl = (sigma_nl * (1.0 + z) / Hz) * h
            else:
                R_nl = 0.0
                
            # Safely calculate mu^2
            mu_sq = np.divide(k_par_sq, k_mag_sq, out=np.zeros_like(k_mag_sq), where=(k_mag_sq != 0))
            
            # Kaiser Factor and FoG
            rsd_factor = (bias + f * mu_sq)**2
            D2_fog = np.exp(-k_par_sq * (R_nl**2)) if sigma_nl > 0 else 1.0
            
            pk_3d = pk_3d * rsd_factor * D2_fog
        else:
            # If no RSD, just apply the isotropic isotropic clustering
            pk_3d = pk_3d * (bias**2)

        # =======================================================
        # 3. Apply Discretization Window (MAS)
        # =======================================================
        if MAS is not None:
            # Fixed: Removed self.N argument to match your earlier definition
            W2_k = self.get_window_correction_grid(method=MAS)
            pk_3d = pk_3d * W2_k

        # =======================================================
        # 4. Bin the 3D Model into 1D k-shells
        # =======================================================
        k_min, k_nyq = self._isotropic_k_range()
        k_bin = k_min
        
        bins = np.arange(k_min, k_nyq + k_bin, k_bin)
        kc = 0.5 * (bins[1:] + bins[:-1])
        
        vals = np.zeros(kc.size)
        stddev = np.zeros(kc.size)
        
        idxs = np.digitize(k_mag.flatten(), bins)
        pk_flat = pk_3d.flatten()
        
        for i in range(1, bins.size):
            ii = (idxs == i)
            if np.any(ii):
                vals[i-1] = np.mean(pk_flat[ii])
                stddev[i-1] = np.std(pk_flat[ii]) / np.sqrt(np.sum(ii))
            else:
                vals[i-1] = np.nan
                stddev[i-1] = np.nan
        print('DONE')
        
        return kc, vals, stddev

    def model_obs_power_CC(self, km, pkm, bias_HI, bias_gal, Tb, sigdeg, MAS='NGP', rsd=True, sigma_nl=120, fog_model='lorentzian'):
        """
        Computes the theoretical Cross-Correlation power spectrum by forward-modeling 
        the matter power spectrum on the 3D FFT grid.
        
        Parameters:
        -----------
        km, pkm : array_like
            The 1D theoretical matter power spectrum (wavenumbers and power).
        bias_HI : float
            The linear bias factor of the Intensity Mapping tracer.
        bias_gal : float
            The linear bias factor of the Galaxy tracer.
        Tb : float
            The mean brightness temperature (signal amplitude).
        sigdeg : float
            Standard deviation of the Gaussian beam in degrees.
        MAS : 'NGP' or 'CIC' or None
            Mass assignment method used.
        rsd : Bool
            Set True to apply Kaiser and FoG effects.
        sigma_nl : float
            Velocity dispersion in km/s for Finger of God damping.
        fog_model : {'lorentzian', 'gaussian', 'none'}
            Power-level Fingers-of-God damping convention. ``lorentzian`` is
            ``[1 + (k_parallel sigma_v/(a H))^2]^-1`` and matches
            :meth:`CosmoBox.linear_rsd_density`.
        """
        import pyccl as ccl
        import numpy as np
        
        # =======================================================
        # 1. Calculate Physical Scales & Cosmology
        # =======================================================
        z = self.box.redshift
        scale_factor = 1.0 / (1.0 + z)
        h = self.box.cosmo['h']
        Hz = 100. * h * ccl.h_over_h0(self.box.cosmo, scale_factor)
        
        # Transverse beam scale in Mpc. The FastBox FFT grid and CCL
        # comoving distances are both expressed in Mpc, so no h conversion is
        # applied here.
        if sigdeg > 0:
            chi_mpc = ccl.comoving_radial_distance(self.box.cosmo, scale_factor)
            R_beam = np.radians(sigdeg) * chi_mpc
        else:
            R_beam = 0.0

        # Calculate RSD parameters. This must match CosmoBox.linear_rsd_density.
        if rsd:
            f = ccl.growth_rate(self.box.cosmo, scale_factor)
            R_nl = sigma_nl / (scale_factor * Hz) if sigma_nl > 0 else 0.0
            fog_model = fog_model.lower()
            if fog_model not in {'lorentzian', 'gaussian', 'none'}:
                raise ValueError("fog_model must be 'lorentzian', 'gaussian', or 'none'")

        # =======================================================
        # 2. Create the exact 3D Fourier Grid
        # =======================================================
        kx = 2.0 * np.pi * np.fft.fftfreq(self.box.Nx, d=self.box.Lx / self.box.Nx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.box.Ny, d=self.box.Ly / self.box.Ny)
        kz = 2.0 * np.pi * np.fft.fftfreq(self.box.Nz, d=self.box.Lz / self.box.Nz)
        
        kx3d = kx[:, None, None]
        ky3d = ky[None, :, None]
        kz3d = kz[None, None, :]
        
        k_perp_sq = kx3d**2 + ky3d**2
        k_par_sq = kz3d**2
        k_mag_sq = k_perp_sq + k_par_sq
        k_mag = np.sqrt(k_mag_sq)
        
        # =======================================================
        # 3. Evaluate 3D Theory & Apply RSD + Damping Factors
        # =======================================================
        pk_3d = np.interp(k_mag.flatten(), km, pkm).reshape(k_mag.shape)
        
        # --- RSD (Kaiser + FoG) ---
        if rsd:
            mu_sq = np.divide(k_par_sq, k_mag_sq, out=np.zeros_like(k_mag_sq), where=(k_mag_sq != 0))
            
            # Cross-Correlation Kaiser Factor: (b_HI + f*mu^2) * (b_gal + f*mu^2)
            rsd_factor = (bias_HI + f * mu_sq) * (bias_gal + f * mu_sq)
            
            if fog_model == 'lorentzian' and sigma_nl > 0:
                D2_fog = 1.0 / (1.0 + k_par_sq * R_nl**2)
            elif fog_model == 'gaussian' and sigma_nl > 0:
                D2_fog = np.exp(-k_par_sq * R_nl**2)
            else:
                D2_fog = 1.0
                
            pk_3d = pk_3d * rsd_factor * D2_fog
        else:
            # No RSD: just use isotropic cross-bias
            pk_3d = pk_3d * (bias_HI * bias_gal)

        # Apply single Temperature scaling for Cross-Correlation
        pk_3d = pk_3d * Tb
        
        # --- Observational Damping ---
        D2_beam = np.exp(-k_perp_sq * (R_beam**2)) if sigdeg > 0 else 1.0
  
        # Cross-power uses exactly one power of the beam (sqrt(D2_beam) = B_beam)
        pk_3d_obs = pk_3d * np.sqrt(D2_beam)
        
        # Discretization Window
        if MAS is not None:
            # Retain TWO powers of pixelization for Cross-Correlation, applied directly to pk_3d_obs
            W2_k = self.get_window_correction_grid(method=MAS)
            pk_3d_obs = pk_3d_obs * np.sqrt(W2_k)

        # =======================================================
        # 4. Bin the 3D Model into 1D k-shells
        # =======================================================
        k_min, k_nyq = self._isotropic_k_range()
        k_bin = k_min
        
        bins = np.arange(k_min, k_nyq + k_bin, k_bin)
        kc = 0.5 * (bins[1:] + bins[:-1])
        
        vals = np.zeros(kc.size)
        stddev = np.zeros(kc.size)
        
        idxs = np.digitize(k_mag.flatten(), bins)
        pk_flat = pk_3d_obs.flatten()
        
        for i in range(1, bins.size):
            ii = (idxs == i)
            if np.any(ii):
                vals[i-1] = np.mean(pk_flat[ii])
                stddev[i-1] = np.std(pk_flat[ii]) / np.sqrt(np.sum(ii))
            else:
                vals[i-1] = np.nan
                stddev[i-1] = np.nan
        print('DONE')
        
        return kc, vals, stddev 




    
        
    def matter_power_spectrum(self, k, rsd=False, sigma_nl=0):
        """
        Calculate the theoretical nonlinear power spectrum for the given 
        cosmological parameters, using CCL. Does not depend on the realisation.
        
        Parameters:
        -----------
        k : array_like
            k values to evaluate power spectrum
        rsd : Bool
            Set True to consider Kaiser and FoG
        sigma_nl : float
            non-linear velocity. Set 0 to not consider FoG
            
        Returns:
        --------
            k, pk (array_like):
                Wavenumbers, from 10^-3.5 to 10^1, in Mpc^-1, and the 
                theoretical nonlinear power spectrum, in (Mpc)^3.
        """

        
        # 1. Calculate the real-space matter power spectrum P_m(k)
        pk = ccl.nonlin_matter_power(self.box.cosmo, k=k, a=self.box.scale_factor)
        
        if rsd is False:
            return k, pk
            
        elif rsd is True:
            # 2. Cosmology & Redshift parameters
            z = (1.0 / self.box.scale_factor) - 1.0
            
            # Linear growth rate (f)
            f = ccl.growth_rate(self.box.cosmo, self.box.scale_factor)
            
            # Hubble parameter at z in km/s/Mpc
            Hz = 100. * self.box.cosmo['h'] * ccl.h_over_h0(self.box.cosmo, self.box.scale_factor)
            
            # 3. Finger of God (FoG) Scale
            # Since k is in Mpc^-1, R_nl must be in Mpc
            if sigma_nl > 0:
                R_nl = (sigma_nl * (1.0 + z)) / Hz
            else:
                R_nl = 0.0
                
            # 4. Spherically Average over mu (angle to the line of sight)
            n_mu = 200
            mu = np.linspace(0, 1, n_mu)
            
            # Create a 2D grid for k and mu combinations
            k_grid, mu_grid = np.meshgrid(k, mu, indexing='ij')
            
            # Broadcast 1D P_m(k) to the 2D grid shape: (len(k), len(mu))
            pk_grid = pk[:, None]
            
            # Kaiser effect: (b + f*mu^2)^2. For pure matter, bias b = 1.0
            kaiser_factor = (1.0 + f * (mu_grid**2))**2
            
            # FoG Damping: exp(-(k * mu * R_nl)^2)
            if sigma_nl > 0:
                fog_damping = np.exp(-((k_grid * mu_grid * R_nl)**2))
            else:
                fog_damping = 1.0
                
            # Combine to get the 2D Redshift-Space Power Spectrum
            pk_2d = pk_grid * kaiser_factor * fog_damping
            
            # Integrate (average) over mu to get the 1D spherically averaged P(k)
            pk_1d = np.trapz(pk_2d, x=mu, axis=1)
            
            return k, pk_1d

    def est_2pcf(self, cube):
        """
        Calculates the exact 1D two-point correlation function (2PCF) from a 3D cube 
        using the Wiener-Khinchin theorem and Fast Fourier Transforms.
        For now, just use it for HI signal, since there is no MAS correction
        for the galaxy mesh yet
        
        Parameters:
        -----------
        cube : ndarray
            The 3D data cube (e.g., temperature map or galaxy counts).
        box : object
            The simulation box object containing dimensions (Lx, Ly, Lz) and grid size (N).
            
        Returns:
        --------
        r_c, xi_1d : ndarray
            The binned radial distances, 2PCF values, and errors on the mean.
        """
        # 1. Force the mean to zero to isolate the fluctuations (clustering)
        # This prevents the massive k=0 mode from destroying the signal.
        delta_cube = cube - np.mean(cube)
        
        # 2. Forward FFT to get the 3D density in Fourier space
        delta_k = np.fft.fftn(delta_cube)
        
        # 3. The 3D Power Spectrum (magnitude squared)
        pk_3d = np.abs(delta_k)**2
        
        # 4. Inverse FFT to get the 3D Correlation Function
        # We divide by N^3 to enforce the correct normalization so that xi(r=0) = variance
        xi_3d = np.fft.ifftn(pk_3d).real / (self.box.N**3)
        
        # 5. Create the 3D distance grid (handles periodic wrap-around automatically)
        x = np.fft.fftfreq(self.box.N, d=1.0/self.box.Lx)
        y = np.fft.fftfreq(self.box.N, d=1.0/self.box.Ly)
        z = np.fft.fftfreq(self.box.N, d=1.0/self.box.Lz)
        
        x3d = x[:, None, None]
        y3d = y[None, :, None]
        z3d = z[None, None, :]
        
        r_3d = np.sqrt(x3d**2 + y3d**2 + z3d**2)
        
        # 6. Optimal Binning based on Grid Resolution
        # Min radius is dx, Max radius is half the box
        dx = self.box.Lx / self.box.N
        dy = self.box.Ly / self.box.N
        dz = self.box.Lz / self.box.N
        r_min = min(dx, dy, dz)
        r_max = min(self.box.Lx, self.box.Ly, self.box.Lz) / 2.0
        
        # Bins spaced exactly by r_min
        bins = np.arange(r_min, r_max + r_min, r_min)
        r_c = 0.5 * (bins[1:] + bins[:-1])
        
        xi_1d = np.zeros(r_c.size)
        
        idxs = np.digitize(r_3d.flatten(), bins)
        xi_flat = xi_3d.flatten()
        
        # 7. Spherical Averaging
        for i in range(1, bins.size):
            ii = (idxs == i)
            if np.any(ii):
                xi_1d[i-1] = np.mean(xi_flat[ii])
            else:
                xi_1d[i-1] = np.nan
                
        return r_c, xi_1d


    def bispectrum_equilateral(self, cube, MAS=None, n_bins=8):
        """
        Pure numpy implementation of the FFT-based Equilateral Bispectrum estimator.
        Calculates the 3-point clustering for triangles where k1 = k2 = k3 = k.
        Uses Scoccimarro estimator (https://arxiv.org/abs/astro-ph/0004086)
        
        Parameters:
        -----------
        cube : ndarray
            The 3D data cube (e.g., temperature map or galaxy density).
        box : object
            The simulation box object containing dimensions (Lx, Ly, Lz) and grid size (N).
        MAS : str or None
            Mass assignment scheme to deconvolve ('NGP', 'CIC', None).
        n_bins : int
            Number of k-bins. (Fewer/wider bins are better for the bispectrum to 
            ensure enough closed triangles can form inside the shell).
            
        Returns:
        --------
        k_c, B_eq, triangles : ndarrays
            The bin centers, the Equilateral Bispectrum B(k,k,k), and the 
            number of closed triangles found in each bin.
        """
        # 1. Enforce zero mean to isolate the fluctuations
        delta = cube - np.mean(cube)
        
        # 2. Forward FFT
        delta_k = np.fft.fftn(delta)
        delta_k[0, 0, 0] = 0.0  # Suppress DC leakage
        
        # 3. Create the exact 3D k-grid
        kx = np.fft.fftfreq(self.box.N, d=self.box.Lx/self.box.N) * 2 * np.pi
        ky = np.fft.fftfreq(self.box.N, d=self.box.Ly/self.box.N) * 2 * np.pi
        kz = np.fft.fftfreq(self.box.N, d=self.box.Lz/self.box.N) * 2 * np.pi
        
        kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)
        
        # 4. MAS Window Deconvolution
        if MAS is not None:
            assign_dict = {'NGP': 1, 'CIC': 2}
            p = assign_dict.get(MAS.upper(), 0)
            
            if p > 0:
                W = np.sinc(np.fft.fftfreq(self.box.N))
                Wx = W[:, None, None]
                Wy = W[None, :, None]
                Wz = W[None, None, :]
                
                # For the Bispectrum, we deconvolve the field ITSELF (W^p)
                # rather than the power (W^2p).
                W_k = (Wx * Wy * Wz)**p
                W_k[W_k == 0] = np.inf  # Protect against division by zero at nyquist edges
                delta_k /= W_k
                
        # 5. Set up logarithmic or linear k-bins
        k_f, k_nyq = self._isotropic_k_range()
        
        # Bispectrum requires wider bins than P(k) to form closed triangles natively
        bins = np.linspace(k_f, k_nyq, n_bins + 1)
        k_c = 0.5 * (bins[1:] + bins[:-1])
        
        B_eq = np.zeros(n_bins)
        triangles = np.zeros(n_bins)
        V_box = self.box.Lx * self.box.Ly * self.box.Lz
        
        # 6. The Scoccimarro FFT Estimator Loop
        for i in range(n_bins):
            # Create a boolean filter mask for the current k-shell
            mask = (k_mag >= bins[i]) & (k_mag < bins[i+1])
            
            if not np.any(mask):
                B_eq[i] = np.nan
                continue
                
            # Filter the complex Fourier field
            delta_k_shell = delta_k * mask
            
            # Inverse FFT the field AND the mask back to real space.
            # (.real drops the ~1e-16 floating point imaginary residuals)
            delta_x_shell = np.fft.ifftn(delta_k_shell).real
            I_x_shell = np.fft.ifftn(mask).real
            
            # The magic volume integrals (just array sums in discrete numpy!)
            num = np.sum(delta_x_shell**3)
            den = np.sum(I_x_shell**3)
            
            if den > 0:
                # The exact cosmological normalization factor: V^2 / N^9
                B_eq[i] = (V_box**2 / (self.box.N**9)) * (num / den)
                # The denominator mathematically counts the number of closed triangles / N^6
                triangles[i] = den * (self.box.N**6) 
            else:
                B_eq[i] = np.nan
                
        return k_c, B_eq, triangles
