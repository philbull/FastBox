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
        
        Parameters:
            box (CosmoBox):
                Object containing a simulation box.
        """
        self.box = box
    
    
    def signal_amplitude(self, amp, redshift):
        """
        Overall signal amplitude (e.g. mean brightness temperature). This is a 
        simple constant-amplitude model.
        
        Parameters:
            amp (float):
                Overall amplitude.
            
            redshift (float):
                Redshift to evaluate the amplitude at.
        
        Returns:
            bias (float):
                Bias at given redshift.
        """
        return amp + 0.*redshift # same shape as redshift
    
    
    def linear_bias(self, b0, redshift):
        """
        Linear bias model, b(z) = b0 sqrt(1 + z)
        
        Parameters:
            b0 (float):
                Overall bias amplitude.
        
        redshift (float):
            Redshift to evaluate the bias at.
        
        Returns:
            bias (float):
                Bias at given redshift.
        """
        return b0 * np.sqrt(1. + redshift)



class HITracer(TracerModel):
    
    def __init__(self, box, OmegaHI0=0.000486, bHI0=0.677105):
        """
        An object to manage a biased tracer on top of a realisation of a 
        density field in a box.
        
        Parameters:
            box (CosmoBox):
                Object containing a simulation box.
        
            OmegaHI0 (float, optional):
                Fractional density of HI at redshift 0.
                
            bHI0 (float, optional):
                HI bias at redshift 0.
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
        
        Parameters:
            redshift (float, optional):
                Central redshift to evaluate the signal amplitude at. If not 
                specified, uses `self.box.redshift`.
            
            formula (str, optional):
                Which fitting formula to use for the brightness temperature. Some 
                of the options are a function of Omega_HI(z)
                
                - ``powerlaw``: Simple power-law fit to Mario's updated data 
                (powerlaw M_HI function with alpha=0.6) (Default)
                
                - ``hall``: From Hall, Bonvin, and Challinor.
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
        
        Parameters:
            redshift (float, optional):
                Central redshift to evaluate the signal amplitude at. If not 
                specified, uses `self.box.redshift`.
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
        
        Parameters:
            redshift (float, optional):
                Central redshift to evaluate the signal amplitude at. If not 
                specified, uses `self.box.redshift`.
        """
        if redshift is None:
            redshift = self.box.redshift
        z = redshift
        
        # Fitting formula; see Bull et al. (2015)
        return (self.OmegaHI0 / 0.000486) \
             * (4.8304e-04 + 3.8856e-04*z - 6.5119e-05*z**2.)




class GalaxyTracer:
    def __init__(self, box, vol_density, bias=1.0):
        """
        Initialise the Galaxy Tracer.
        
        Parameters:
        -----------
        box : fastbox.Box
            The FastBox instance containing the density field and grid parameters.
        vol_density : float
            The mean volumetric density of galaxies in Mpc^-3.
        bias : float
            The linear galaxy bias factor.
        """
        self.box = box
        self.vol_density = vol_density
        self.bias = bias

    def generate_catalogue(self, density_field):
        """
        Generates a discrete galaxy catalogue via Poisson sampling.
        """
        voxel_vol = (self.box.Lx * self.box.Ly * self.box.Lz) / \
                    (self.box.Nx * self.box.Ny * self.box.Nz)
        
        expected_gals = self.vol_density * voxel_vol * (1.0 +  density_field)
        expected_gals[expected_gals < 0] = 0.0 
        
        N_gals_per_voxel = np.random.poisson(expected_gals)
        
        voxel_indices = np.where(N_gals_per_voxel > 0)
        counts = N_gals_per_voxel[voxel_indices]
        
        dx = self.box.Lx / self.box.Nx
        dy = self.box.Ly / self.box.Ny
        dz = self.box.Lz / self.box.Nz
        
        x_base = voxel_indices[0] * dx
        y_base = voxel_indices[1] * dy
        z_base = voxel_indices[2] * dz
        
        x_base_exp = np.repeat(x_base, counts)
        y_base_exp = np.repeat(y_base, counts)
        z_base_exp = np.repeat(z_base, counts)
        
        total_gals = np.sum(counts)
        x_gal = x_base_exp + np.random.uniform(0, dx, total_gals)
        y_gal = y_base_exp + np.random.uniform(0, dy, total_gals)
        z_gal = z_base_exp + np.random.uniform(0, dz, total_gals)
        
        galaxy_positions = np.vstack((x_gal, y_gal, z_gal)).T
        
        return galaxy_positions

    def generate_mesh(self, density_field, method='NGP',overdensity=False):
        """
        Generates a discrete galaxy catalogue and assigns it to a density mesh.
        """
        positions = self.generate_catalogue(density_field)
        
        method = method.upper()
        if method == 'NGP':
            return self._assign_ngp(positions,overdensity)
        elif method == 'CIC':
            return self._assign_cic(positions,overdensity)
        else:
            raise ValueError(f"Mass assignment method '{method}' not recognised. Use 'NGP' or 'CIC'.")

    def _assign_ngp(self, positions, overdensity=False):
        """Internal method for Nearest Grid Point mass assignment."""
        edges_x = np.linspace(0, self.box.Lx, self.box.Nx + 1)
        edges_y = np.linspace(0, self.box.Ly, self.box.Ny + 1)
        edges_z = np.linspace(0, self.box.Lz, self.box.Nz + 1)
        
        pos = np.asarray(positions)
        x = pos[:, 0] % self.box.Lx
        y = pos[:, 1] % self.box.Ly
        z = pos[:, 2] % self.box.Lz
        
        grid, _ = np.histogramdd((x, y, z), bins=(edges_x, edges_y, edges_z))

        if overdensity==True:
            mean_counts = np.mean(grid)
            grid = (grid / mean_counts) - 1.0
        
        return grid
    
    def _assign_cic(self, positions, overdensity=False):
        """Internal method for Cloud-in-Cell mass assignment."""
        pos = np.asarray(positions)
        N_total_cells = self.box.Nx * self.box.Ny * self.box.Nz
        
        dx = self.box.Lx / self.box.Nx
        dy = self.box.Ly / self.box.Ny
        dz = self.box.Lz / self.box.Nz
    
        u = (pos[:, 0] / dx - 0.5) % self.box.Nx
        v = (pos[:, 1] / dy - 0.5) % self.box.Ny
        w = (pos[:, 2] / dz - 0.5) % self.box.Nz
    
        i0 = np.floor(u).astype(int)
        j0 = np.floor(v).astype(int)
        k0 = np.floor(w).astype(int)
    
        i1 = (i0 + 1) % self.box.Nx
        j1 = (j0 + 1) % self.box.Ny
        k1 = (k0 + 1) % self.box.Nz
    
        dwx = u - i0
        dwy = v - j0
        dwz = w - k0
    
        twx = 1.0 - dwx
        twy = 1.0 - dwy
        twz = 1.0 - dwz
    
        w000 = twx * twy * twz
        w100 = dwx * twy * twz
        w010 = twx * dwy * twz
        w110 = dwx * dwy * twz
        w001 = twx * twy * dwz
        w101 = dwx * twy * dwz
        w011 = twx * dwy * dwz
        w111 = dwx * dwy * dwz
    
        def flat_idx(i, j, k):
            return i * (self.box.Ny * self.box.Nz) + j * self.box.Nz + k
    
        grid_flat = (
            np.bincount(flat_idx(i0, j0, k0), weights=w000, minlength=N_total_cells) +
            np.bincount(flat_idx(i1, j0, k0), weights=w100, minlength=N_total_cells) +
            np.bincount(flat_idx(i0, j1, k0), weights=w010, minlength=N_total_cells) +
            np.bincount(flat_idx(i1, j1, k0), weights=w110, minlength=N_total_cells) +
            np.bincount(flat_idx(i0, j0, k1), weights=w001, minlength=N_total_cells) +
            np.bincount(flat_idx(i1, j0, k1), weights=w101, minlength=N_total_cells) +
            np.bincount(flat_idx(i0, j1, k1), weights=w011, minlength=N_total_cells) +
            np.bincount(flat_idx(i1, j1, k1), weights=w111, minlength=N_total_cells)
        )
    
        grid = grid_flat.reshape((self.box.Nx, self.box.Ny, self.box.Nz))
        if overdensity==True:
            mean_counts = np.mean(grid)
            grid = (grid / mean_counts) - 1.0
            
        return grid

    def apply_rsd(self, positions, sigma_nl=120.0):
            """
            Applies Redshift Space Distortions (RSD) to discrete galaxy coordinates.
            Calculates both the coherent large-scale infall (Kaiser effect) and 
            non-linear dispersion (Fingers of God).
            
            Parameters:
            -----------
            positions : ndarray
                (N, 3) array of real-space galaxy coordinates.
            sigma_nl : float
                Velocity dispersion for FoG in km/s. Set to 0 to disable FoG.
                
            Returns:
            --------
            positions_rsd : ndarray
                (N, 3) array of redshift-space galaxy coordinates.
            """
            positions_rsd = np.copy(positions)
            N_gals = positions_rsd.shape[0]
            
            # 1. Calculate Expansion Rate H(z) in km/s/Mpc
            Hz = 100. * self.box.cosmo['h'] * ccl.h_over_h0(self.box.cosmo, self.box.scale_factor)
            
            # Array to accumulate total line-of-sight velocity for each galaxy (in km/s)
            vel_shift_kms = np.zeros(N_gals)
            
            # 2. Kaiser Effect (Coherent velocities from linear theory)
            # Generate Fourier-space velocity field
            vel_k = self.box.realise_velocity(delta_x=self.box.delta_x, inplace=True)
            # Inverse FFT to get real-space radial velocity (z-direction)
            vel_z = fft.ifftn(vel_k[2]).real
            
            # Map galaxies to their nearest grid cells to extract local velocity
            dx = self.box.Lx / self.box.Nx
            dy = self.box.Ly / self.box.Ny
            dz = self.box.Lz / self.box.Nz
            
            i = np.floor((positions[:, 0] % self.box.Lx) / dx).astype(int)
            j = np.floor((positions[:, 1] % self.box.Ly) / dy).astype(int)
            k = np.floor((positions[:, 2] % self.box.Lz) / dz).astype(int)
            
            vel_shift_kms += vel_z[i, j, k]
                
            # 3. Fingers of God (Incoherent thermal velocities)
            if sigma_nl > 0.0:
                vel_shift_kms += sigma_nl * np.random.normal(loc=0.0, scale=1.0, size=N_gals)
                
            # 4. Apply Spatial Displacement to Z-Coordinates
            # Shift in comoving Mpc = velocity / H(z)
            dz_mpc = vel_shift_kms / Hz
            positions_rsd[:, 2] = (positions_rsd[:, 2] + dz_mpc) % self.box.Lz
            
            return positions_rsd




