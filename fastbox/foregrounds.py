"""
Add foreground emission to datacubes.
"""
import numpy as np
import pyccl as ccl
import pylab as plt
from numpy import fft
import scipy.ndimage
from functools import partial
import os, warnings

# Load optional modules
try:
    import healpy as hp
except:
    warnings.warn("Module `healpy` not found. Some functions in "    
                  "fastbox.foregrounds will not work", 
                  ImportWarning)

# Physical constants
KBOLTZ = 1.3806488e-23 
C_LIGHT = 2.99792458e8 # m/s
H_PLANCK = 6.626e-34
CMB_TEMP = 2.73 # K

# Default paths to Planck simulation files, used by PlanckSkyModel
DEFAULT_PLANCK_SIM_PATHS = {
  'ff217':   'fastbox/COM_SimMap_freefree-ffp10-skyinbands-217_2048_R3.00_full.fits',
  'sync217': 'fastbox/COM_SimMap_synchrotron-ffp10-skyinbands-217_2048_R3.00_full.fits',
  'sync353': 'fastbox/COM_SimMap_synchrotron-ffp10-skyinbands-353_2048_R3.00_full.fits'
}


class ForegroundModel(object):
    
    def __init__(self, box):
        """
        An object to manage the addition of foregrounds on top of a realisation 
        of a density field in a box.
        
        Parameters:
            box (CosmoBox):
                Object containing a simulation box.
        """
        self.box = box
        
    
    def realise_foreground_amp(self, amp, beta, monopole, smoothing_scale=None, 
                               redshift=None):
        """
        Create realisation of the matter power spectrum by randomly sampling 
        from Gaussian distributions of variance P(k) for each k mode.
        
        Parameters:
            amp (float):
                Amplitude of foreground power spectrum, in units of the field 
                squared (e.g. if the field is in units of mK, this should be in 
                [mK]^2).
            
            beta (float):
                Angular power-law index.
            
            monopole (float):
                Zero-point offset of the foreground at the reference frequency. 
                Should be in the same units as the field, e.g. mK.
            
            smoothing_scale (float, optional):
                Additional angular smoothing scale, in degrees. Default: None 
                (no smoothing). 
                
            redshift (float, optional):
                Redshift to evaluate the centre of the box at. Default: Same value 
                as `self.box.redshift`.
        """
        # Check redshift
        if redshift is None:
            redshift = self.box.redshift
        scale_factor = 1. / (1. + redshift)
        
        # Calculate comoving distance to box redshift
        r = ccl.comoving_angular_distance(self.box.cosmo, scale_factor)
        
        # Angular Fourier modes (in 2D)
        k_perp = 2.*np.pi * np.sqrt(  (self.box.Kx[:,:,0]/self.box.Lx)**2. 
                                    + (self.box.Ky[:,:,0]/self.box.Ly)**2.)
        
        # Foreground angular power spectrum
        # \ell ~ k_perp r / 2
        # Use FG power spectrum model from Santos et al. (2005)
        C_ell = amp * (0.5*k_perp*r / 1000.)**(beta)
        C_ell[np.isinf(C_ell)] = 0. # Remove inf at k=0
        
        # Normalise the power spectrum properly (factor of area, and norm. 
        # factor of 2D DFT)
        C_ell *= (self.box.N**4.) / (self.box.Lx * self.box.Ly)
        
        # Generate Gaussian random field with given power spectrum
        re = np.random.normal(0.0, 1.0, np.shape(k_perp))
        im = np.random.normal(0.0, 1.0, np.shape(k_perp))
        fg_k = (re + 1.j*im) * np.sqrt(C_ell) # factor of 2x larger variance
        fg_k[k_perp==0.] = 0. # remove zero mode
        
        # Transform to real space. Discarding the imag part fixes the extra 
        # factor of 2x in variance from above
        fg_x = fft.ifftn(fg_k).real + monopole
        
        # Apply angular smoothing
        if smoothing_scale is not None:
            ang_x, ang_y = self.box.pixel_array(redshift=redshift)
            sigma = smoothing_scale / (ang_x[1] - ang_x[0])
            fg_x = scipy.ndimage.gaussian_filter(fg_x, sigma=sigma, mode='wrap')
        
        return fg_x
    
    
    def realise_spectral_index(self, mean_spec_idx, std_spec_idx, 
                               smoothing_scale, redshift=None):
        """
        Generate a Gaussian random realisation of the spectral index and apply 
        a smoothing scale.
        
        Parameters:
            mean_spec_idx (float):
                Mean value of the spectral index.
            
            std_spec_idx (float):
                Standard deviation of the spectral index.
            
            smoothing_scale (float):
                Angular smoothing scale, in degrees.
            
            redshift (float, optional):
                Redshift to evaluate the centre of the box at. Default: Same value 
                as `self.box.redshift`.
        """
        # Generate uncorrelated Gaussian random field
        alpha = np.random.normal(mean_spec_idx, std_spec_idx, 
                                 self.box.Kx[:,:,0].shape)
        
        # Smooth with isotropic Gaussian
        ang_x, ang_y = self.box.pixel_array(redshift=redshift)
        sigma = smoothing_scale / (ang_x[1] - ang_x[0])
        alpha = scipy.ndimage.gaussian_filter(alpha, sigma=sigma, mode='wrap')
        return alpha
    
    
    def construct_cube(self, amps, spectral_idx, freq_ref=130., redshift=None):
        """
        Construct a foreground datacube from an input 2D amplitude map and 
        spectral index map.
        
        Parameters:
            amps (array_like):
                2D array of amplitudes.
            
            spectral_index (array_like):
                2D array of spectral indices, or float.
            
            freq_ref (float, optional):
                Reference frequency, in MHz.
            
            redshift (float, optional):
                Redshift to evaluate the centre of the box at. Default: Same value 
                as `self.box.redshift`.
        """
        # Get frequency array and scaling
        freqs = self.box.freq_array(redshift=redshift)
        if isinstance(spectral_idx, float):
            ffac = ((freqs / freq_ref)**spectral_idx)[np.newaxis,np.newaxis,:]
        else:
            ffac = (freqs/freq_ref)[np.newaxis,np.newaxis,:]**spectral_idx[:,:,np.newaxis]
        
        # Return datacube
        return amps[:,:,np.newaxis] * ffac
        


class GlobalSkyModel(object):
    
    def __init__(self, box):
        """
        An object to manage the addition of foregrounds from pyGDSM on top of a 
        realisation of a density field in a box. Uses the GlobalSkyModel2016 
        class.
        
        Parameters:
            box (CosmoBox):
                Object containing a simulation box.
        """
        self.box = box
        
        # Check for imports specific to this class
        try:
            from pygdsm import GlobalSkyModel2016
        except:
            print("pygdsm is not installed")
            raise
        
        # Initialise GSM
        self.gsm = GlobalSkyModel2016(freq_unit='MHz')
    
    
    def construct_cube(self, rotation=(0., -62., 0.), redshift=None, loop=True, 
                       verbose=True):
        """
        Construct a foreground datacube from GDSM.
        
        Parameters:
            rotation (tuple, optional):
                Rotation of the field from Galactic coordinates, used by healpy 
                ``gnomview`` when projecting the field.
                
            redshift (float, optional):
                Redshift to evaluate the centre of the box at. Default: Same value 
                as `self.box.redshift`.
            
            loop : bool, optional
                Whether to fetch the GSM maps for all frequency channels at once 
                (False), or to loop through them one by one (True).
            
            verbose : bool, optional
                If True, print status messages.
        """
        import healpy as hp
        
        # Initialise empty cube
        fgcube = np.zeros((self.box.N, self.box.N, self.box.N))
        
        # Get frequency array and scaling
        freqs = self.box.freq_array(redshift=redshift)
        ang_x, ang_y = self.box.pixel_array(redshift=redshift)
        delta_ang_x = np.max(ang_x) - np.min(ang_x)
        delta_ang_y = np.max(ang_y) - np.min(ang_y)
        lon0 = rotation[0]
        lat0 = rotation[1]
        
        # Cartesian projection of maps
        npix = self.box.N
        lonra = [lon0 - 0.5*delta_ang_x, lon0 + 0.5*delta_ang_x]
        latra = [lat0 - 0.5*delta_ang_y, lat0 + 0.5*delta_ang_y]
        proj = hp.projector.CartesianProj(lonra=lonra, latra=latra, coord='G',
                                          xsize=npix, ysize=npix)
        
        # Fetch maps and perform projection
        if loop:
            for i, freq in enumerate(freqs):
                if verbose and i % 10 == 0:
                    print("    Channel %d / %d" % (i, len(freqs)))
                
                # Get map and project to Cartesian grid
                m = self.gsm.generate(freq)
                nside = hp.npix2nside(m.size)
                fgcube[:,:,i] = proj.projmap(m, vec2pix_func=partial(hp.vec2pix, nside))[::-1] 
        else:
            # Fetch all maps in one go
            maps = self.gsm.generate(freqs)
            nside = hp.npix2nside(maps[0].size)
            
            # Do projection one channel at a time
            for i, m in enumerate(maps):
                if verbose and i % 10 == 0:
                    print("    Channel %d / %d" % (i, len(freqs)))
                fgcube[:,:,i] = proj.projmap(m, vec2pix_func=partial(hp.vec2pix, nside))[::-1] 
        
        # Return datacube
        return fgcube



class PointSourceModel(object):
    """
    Make point source maps according to the Battye et al. 2013 recipe.
    """

    def __init__(self, box):
        """
        Make point source maps according to the Battye et al. 2013 recipe.
        Based on Eq. 36 of https://arxiv.org/pdf/1209.0343.pdf; see also p99 of
        https://www.research.manchester.ac.uk/portal/files/67403180/FULL_TEXT.PDF

        Parameters:
            box (CosmoBox):
                Object containing a simulation box.
        """ 
        self.box = box
        
        
    def flux_amplitude(self, sjy):
        """Amplitude factor for the flux scaling of the model."""
        logS = np.log10(sjy)
        gamma = 2.593 \
              + 9.333e-2 * logS \
              - 4.839e-4 * logS**2. \
              + 2.488e-1 * logS**3. \
              + 8.995e-2 * logS**4. \
              + 8.506e-3 * logS**5.
        return 10.**gamma
    
    
    def integ_flux(self, sjy):
        """Empirical point source model for integrated flux."""
        return self.flux_amplitude(sjy) * sjy**(-2.5) * sjy

    
    def poisson_pspec(self, sjy):
        """Empirical point source model for Poisson power spectrum."""
        return self.flux_amplitude(sjy) * sjy**(-2.5) * sjy**(2.0)
    
    
    def number_count(self, sjy):
        """Empirical point source model for source count."""
        return self.flux_amplitude(sjy) * sjy**(-2.5)


    def construct_cube(self, flux_cutoff, beta, delta_beta, 
                       redshift=None, nside=256, rotation=(0., -62., 0.),
                       seed_clustering=None, seed_poisson=None):
        """Make point source emission data cube for a box.
        
        Parameters:
            flux_cutoff (float):
                Maximum flux (cutoff) for the point source model, in Jy.
            
            beta (float):
                Mean spectral index value, beta.
            
            delta_beta (float):
                RMS spectral index fluctuation, delta beta.
            
            redshift (float, optional):
                Redshift to evaluate the model at. Default: None (uses box redshift).
            
            nside (int, optional):
                Healpix NSIDE map resolution parameter; must be a power of 2. 
                Default: 256.
            
            rotation (tuple, optional):
                Rotation of the field from Galactic coordinates, used by healpy 
                ``gnomview`` when projecting the field.
            
            seed_clustering, seed_poisson (int, optional):
                Random seed used by the clustering and Poisson models.
        
        Returns:
            fg_cube, T_ps_mean (array_like):
                Point source temperature data cube and mean temperature.
                
                - ``fg_cube (array_like)``: Point source temperature data cube (in mK).
            
                - ``T_ps_mean (array_like)``: Mean temperature (in mK) of point 
                  source component at each freq.
        """
        # Frequency and angular pixel coordinates for the box
        freqs = self.box.freq_array(redshift=redshift) # MHz
        ang_x, ang_y = self.box.pixel_array(redshift=redshift) # deg
        xside = ang_x.size
        yside = ang_y.size
        lon0 = rotation[0]
        lat0 = rotation[1]
        
        # Resolution parameters
        ell = np.arange(nside*3) + 1.0
        npix = 12 * nside * nside
        pixarea = (np.degrees(4.*np.pi) * 60.) / (npix)
        nfreq = freqs.size
        cfact = C_LIGHT**2 / (2 * KBOLTZ * (1.4e9)**2) * 10.**-26

        # Make point source map at reference freq. (1.4 GHz)
        # Get the mean temperature
        intvals = scipy.integrate.quad(lambda sjy: self.integ_flux(sjy), 0., flux_cutoff)
        T_ps0 = cfact * (intvals[0] - intvals[1])

        # Get the clustering contribution
        np.random.seed(seed_clustering)
        clclust = 1.8e-4 * ell**-1.2 * T_ps0**2
        clustmap = hp.sphtfunc.synfast(clclust, nside, new=True)

        # Get the poisson contribution
        # Under 0.01 Jy, Poisson contributions behave as Gaussians
        cl_poisson_low = np.zeros((len(ell)))
        vals = np.arange(1e-6, 0.01, (0.01 - 1e-6)/len(ell))
        for j, ival in enumerate(vals):
            # FIXME: Use cumtrapz here instead
            intvals = scipy.integrate.quad(lambda sjy: self.poisson_pspec(sjy), 0., ival)
            cl_poisson_low[j] = cfact**2 * (intvals[0] - intvals[1])
        
        np.random.seed(seed_poisson)
        poisson_low_map = hp.sphtfunc.synfast(cl_poisson_low, nside, new=True)
        
        # Over 0.01 Jy, need to inject sources into the sky
        shotmap = np.zeros((npix))
        if flux_cutoff > 0.01:
            for ival in np.arange(0.01, flux_cutoff, (flux_cutoff - 0.01)/10.):
                # N is number of sources per steradian per Jansky
                numbster = scipy.integrate.quad(
                                            lambda sjy: self.number_count(sjy), 
                                            ival - 1e-3, 
                                            ival + 1e-3)[0]
                numbsky = int(4 * np.pi * numbster * ival)
                tempval = cfact * scipy.integrate.quad(
                                            lambda sjy: self.integ_flux(sjy), 
                                            0.01, 
                                            ival)[0] / pixarea
                randind = np.random.choice(range(npix), numbsky)
                shotmap[randind] = tempval
        
        # Sum contributions together to get full template map at reference freq.
        map0 = T_ps0 + poisson_low_map + clustmap + shotmap
        
        # Extract npix by npix projected map (at 1.4 GHz) by using gnomview
        npix = self.box.N
        delta_ang_x = np.max(ang_x) - np.min(ang_x)
        delta_ang_y = np.max(ang_y) - np.min(ang_y)
        lonra = [0.0 - 0.5*delta_ang_x, 0.0 + 0.5*delta_ang_x]
        latra = [0.0 - 0.5*delta_ang_y, 0.0 + 0.5*delta_ang_y]
        proj = hp.projector.CartesianProj(lonra=lonra, latra=latra, coord='G',
                                          xsize=npix, ysize=npix)

        nside = hp.get_nside(map0)  
        map0 = proj.projmap(map0, vec2pix_func=partial(hp.vec2pix, nside))[::-1]
        
        # Generate random realisation of spectral index
        spec_idx_map = np.random.normal(beta, scale=delta_beta**2, size=12*nside*nside)
        
        # Use gnomview to get projected spectral index map
        spidxs = proj.projmap(spec_idx_map, vec2pix_func=partial(hp.vec2pix, nside))[::-1]

        # Scale-up maps to different frequencies
        maps = np.zeros((xside, yside, nfreq))
        maps[:,:,:] = map0[:,:,np.newaxis] \
             * (freqs[np.newaxis,np.newaxis,:]/1400.)**(spidxs[:,:,np.newaxis])
        
        # Mean temperature vs freq.
        T_ps_mean = (T_ps0 * (freqs/1400.)**beta).reshape(nfreq, 1)

        return maps*1e3, T_ps_mean*1e3 # mK



class PlanckSkyModel(object):

    def __init__(self, box, free_idx=-2.1, planck_sim_paths=DEFAULT_PLANCK_SIM_PATHS):
        """
        An object to manage the PSM foreground model.

        Parameters:
            box (CosmoBox):
                Object containing a simulation box.
            
            free_idx (float, optional):
                Spectral index of the free-free component. Default: -2.1.
            
            planck_sim_paths (dict):
                Dict containing paths to Planck simulations used by the sky model. 
                Must have keys 'ff217', 'sync217', 'sync353', which are paths to 
                .fits files containing the free-free and synchrotron temperature 
                maps at 217 / 353 GHz as appropriate. 
                Default: ``fastbox.foregrounds.DEFAULT_PLANCK_SIM_PATHS``
        """
        # Try to import healpy
        try:
            import healpy as hp
        except:
            print("healpy is not installed")
            raise
            
        self.box = box
        
        # Store spectral index parameters
        self.free_idx = free_idx
        
        # Store filename dict
        for key in ['ff217', 'sync217', 'sync353']:
            # Check that key exists
            assert key in planck_sim_paths.keys(), \
                "planck_sim_paths argument is missing compulsory key '%s'" % key
            
            # Check that file exists
            f = planck_sim_paths[key]
            if not os.path.exists(f):
                raise ValueError("Could not find file '%s' for key '%s'" % (f, key))
        self.planck_sim_paths = planck_sim_paths
    

    def planck_corr(self, freq_ghz):
        """Correction factor to convert T_CMB to T_RJ.
        
        Parameters:
            freq_ghz (float):
                Frequency, in GHz.
        
        Returns:
            correction (float):
                Correction factor; divide a T_CMB quantity by this to obtain T_RJ.
        """
        freq = freq_ghz * 1e9 # Hz
        factor = H_PLANCK * freq / (KBOLTZ * CMB_TEMP)
        correction = (np.exp(factor)-1.)**2. / (factor**2. * np.exp(factor))
        return correction
    
    
    def read_planck_sim_maps(self):
        """Read Planck simulation maps for synchrotron and free-free emission.
        
        Reads the Planck simulation maps specified in ``self.planck_sim_paths`` 
        as Healpix maps, and converts from T_CMB to T_RJ units.
        
        Returns:
            free217, sync217, sync353 (array_like):
                Healpix map arrays for each simulation map.
        """
        # Load maps and convert from T_CMB to T_RJ
        free217 = hp.fitsfunc.read_map(self.planck_sim_paths['ff217'], 
                                       field=0, nest=False) \
                / self.planck_corr(217.)
        sync217 = hp.fitsfunc.read_map(self.planck_sim_paths['sync217'], 
                                       field=0, nest=False) \
                / self.planck_corr(217.)
        sync353 = hp.fitsfunc.read_map(self.planck_sim_paths['sync353'], 
                                       field=0, nest=False) \
                / self.planck_corr(353.)
        return free217, sync217, sync353
    
    
    def synch_freefree_maps(self, redshift=None, rotation=(0., -62., 0.), 
                            ref_freq=1000., free_idx=None, seed_syncidx=None):
        """Calculate free-free and synchrotron amplitude and spectral index maps.
        
        Uses a set of Planck simulations to calculate maps of the synchrotron 
        and free-free amplitude, and the synchrotron spectral index, on a 
        square patch of sky.
        
        On small scales (below 5 degrees), the synchrotron spectral index is 
        generated from a random Gaussian distribution.
        
        Parameters:
            redshift (float, optional):
                Redshift to evaluate the model at. Default: None (uses box redshift).
            
            rotation (tuple, optional):
                Rotation of the field from Galactic coordinates, used by healpy 
                ``gnomview`` when projecting the field.
            
            ref_freq (float, optional):
                Reference frequency to evaluate the amplitudes at, in MHz.
            
            free_idx (float, optional):
                If set, replace the default free-free index with this value. 
            
            seed_syncidx (int, optional):
                Random seed to use when generating the small-scale spectral index 
                fluctuations.
        
        Returns:
            sync_amp, free_amp, sync_idx (array_like): Synchrotron and free-free 
                amplitudes and synchrotron spectral index.
            
            - ``sync_amp, free_amp (array_like)``:
                Synchrotron and free-free amplitude maps, evaluated at the 
                reference frequency.
            
            - ``sync_idx (array_like)``:
                Synchrotron spectral index map.
        """
        # Get frequency and angular pixel coords
        ang_x, ang_y = self.box.pixel_array(redshift=redshift)
        xside = len(ang_x)
        yside = len(ang_y)
        lon0 = rotation[0]
        lat0 = rotation[1]

        # Read Planck simulated foreground maps
        free217, sync217, sync353 = self.read_planck_sim_maps()

        # Get rid of unphysical values
        free217[np.where(free217 < 0.)[0]] = np.percentile(free217, 3)
        
        # Get free-free spectral index
        if free_idx is None:
            free_idx = self.free_idx
        
        # Calculate synchrotron spectral index
        sync_idx = np.log(sync353/ sync217) / np.log(353./217.)
        
        # Calculate component amplitudes (freqs -> GHz)
        sync_amp = sync217 * ((ref_freq/1000.)/217.)**(sync_idx)
        free_amp = free217 * ((ref_freq/1000.)/217.)**(free_idx)

        # Fill-in small-scale structure for synch spectral index map (which is 
        # MAMD's map at 5 deg)
        ells = np.arange(1., 4001.)
        cl5deg = hp.sphtfunc.anafast(
                        np.random.normal(0.0, np.std(sync_idx), 12*2048*2048), 
                        lmax=4000)
        
        # Synchrotron angular power spectrum
        # Taken from https://arxiv.org/pdf/astro-ph/0408515.pdf
        cls = cl5deg[0] * (1000./ells)**2.4
        np.random.seed(seed_syncidx)
        sync_idx = sync_idx + hp.sphtfunc.synfast(cls, 2048)
        
        # Cartesian projection of maps
        npix = self.box.N
        delta_ang_x = np.max(ang_x) - np.min(ang_x)
        delta_ang_y = np.max(ang_y) - np.min(ang_y)
        lonra = [lon0 - 0.5*delta_ang_x, lon0 + 0.5*delta_ang_x]
        latra = [lat0 - 0.5*delta_ang_y, lat0 + 0.5*delta_ang_y]

        proj = hp.projector.CartesianProj(lonra=lonra, latra=latra, coord='G',
                                          xsize=npix, ysize=npix)

        nside = hp.get_nside(sync_idx)  
        synca = proj.projmap(sync_amp, vec2pix_func=partial(hp.vec2pix, nside))[::-1]
        freea = proj.projmap(free_amp, vec2pix_func=partial(hp.vec2pix, nside))[::-1]
        syncind = proj.projmap(sync_idx, vec2pix_func=partial(hp.vec2pix, nside))[::-1]
        
        # Return amplitudes in mK, and synch. spectral index map
        return synca*1e3, freea*1e3, syncind
    
    
    def construct_cube(self, redshift=None, rotation=(0.,-62.,0.), 
                       ref_freq=1000., free_idx=None, seed_syncidx=None):
        """Make Planck Sky Model (synchrotron + free-free) data cube for a box.
        
        Uses the PSM simulations, with power-law synchrotron emission with a 
        spatially-varying spectral index, and free-free emission with a fixed 
        spectral index.
        
        Parameters:
            redshift (float, optional):
                Redshift to evaluate the model at. Default: None (uses box redshift).
            
            rotation (tuple, optional):
                Rotation of the field from Galactic coordinates, used by healpy 
                ``gnomview`` when projecting the field.
            
            ref_freq (float, optional):
                Reference frequency to evaluate the amplitudes at, in MHz. 
            
            seed_syncidx (int, optional):
                Random seed to use when generating the small-scale spectral index 
                fluctuations.
        
        Returns:
            fg_cube (array_like):
                Planck Sky Model temperature data cube (in mK).
        """
        # Get frequency array
        freqs = self.box.freq_array(redshift=redshift) # MHz
        x = freqs / ref_freq # dimensionless freq.
        
        # Calculate synchrotron/free-free amplitude and spectral idx maps
        sync_amp, free_amp, sync_idx \
                = self.synch_freefree_maps(redshift=redshift, 
                                           rotation=rotation, 
                                           ref_freq=ref_freq,
                                           free_idx=None,
                                           seed_syncidx=seed_syncidx)
        
        # Construct component datacubes and return
        fg_map = sync_amp[:,:,np.newaxis] \
                 * x[np.newaxis,np.newaxis,:]**sync_idx[:,:,np.newaxis] \
               + free_amp[:,:,np.newaxis] \
                 * x[np.newaxis,np.newaxis,:]**self.free_idx
        return fg_map
        
