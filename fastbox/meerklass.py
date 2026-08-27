
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyccl as ccl
from scipy.ndimage import gaussian_filter

import fastbox
from fastbox.foregrounds import ForegroundModel
from fastbox.power import Power

LINE_FREQ_MHZ = 1420.405752
C_LIGHT_M_S = 299792458.0
DISH_DIAMETER_M = 13.5

# Supplied foreground prescription, all temperatures in mK.
FOREGROUND_PARAMETERS = {
    'amplitude': 700.0,
    'angular_power_index': -2.4,
    'spectral_index_mean': -2.80,
    'spectral_index_std': 0.10,
    'spectral_smoothing_deg': 0.9,
    'haslam_monopole_mK_at_408MHz': 30e3,
    'haslam_spectral_index': -2.8,
}

# Values are explicit assumptions, not a measurement from the released map.
FIDUCIAL_PARAMETERS = {
    'Omega_HI': 0.00086,
    'b_HI': 1.13,
    'b_gal': 1.90,
    'r_HI_gal': 0.90,
    'sigma_v_km_s': 120.0,
    'fog_model': 'lorentzian',
    'pca_modes': 4,
    'kmax_mpc_inverse': 0.20,
    'reconvolution_gamma': 1.4,
}


def load_geometry(path: Path) -> dict[str, Any]:
    """Load the geometry file written by ``build_meerklass_geometry.py``."""
    geometry = json.loads(path.read_text())
    required = {'fastbox_shape_x_y_z', 'box_scale_mpc', 'central_redshift',
                'science_band_native_mhz', 'cosmology'}
    absent = required.difference(geometry)
    if absent:
        raise ValueError(f'Geometry file lacks required fields: {sorted(absent)}')
    return geometry


def load_released_window(
    geometry: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load native frequencies, footprint, normalised weights, Tsky, and raw hits.

    Returns
    -------
    frequency : ndarray, shape (Nz,)
        Native channel-centre frequencies in MHz.
    mask_xy : ndarray[bool], shape (Nx, Ny)
        Angular analysis footprint.
    estimator_weights_xyz : ndarray, shape (Nx, Ny, Nz)
        Normalised hit weights for PCA/power-spectrum weighting.
    map_xyz_mK : ndarray, shape (Nx, Ny, Nz)
        Released Tsky map in mK.
    raw_hits_xyz : ndarray, shape (Nx, Ny, Nz)
        Unnormalised two-second hit counts for the radiometer equation.
    """
    path = Path(geometry['source_file'])
    nu_min, nu_max = geometry['science_band_native_mhz']

    with np.load(path, allow_pickle=False) as release:
        frequency = release['nu'].astype(float)
        select = (frequency >= nu_min - 1e-9) & (frequency <= nu_max + 1e-9)

        frequency = frequency[select]
        mask_xy = release['CF_mask'].astype(bool)
        hits_fxy = release['hits'][select].astype(float)

        # The data-release notebook specifies that Tsky is in K.
        released_tsky_mK = 1e3 * release['Tsky'][select].astype(float)

    nx, ny, nz = geometry['fastbox_shape_x_y_z']
    if frequency.size != nz or mask_xy.shape != (nx, ny):
        raise RuntimeError('Released map geometry does not match the generated configuration.')

    # Archive ordering is (frequency, RA, Dec); FastBox uses (RA, Dec, frequency).
    raw_hits_xyz = np.moveaxis(hits_fxy, 0, -1)
    map_xyz_mK = np.moveaxis(released_tsky_mK, 0, -1)

    valid_xyz = (raw_hits_xyz > 0.0) & mask_xy[..., None]
    positive_hits = raw_hits_xyz[valid_xyz]
    if positive_hits.size == 0:
        raise RuntimeError('No positive hit counts lie within the supplied CF mask.')

    # Use only for PCA / weighted-power estimation, not for thermal noise.
    estimator_weights_xyz = np.where(
        valid_xyz,
        raw_hits_xyz / np.median(positive_hits),
        0.0,
    )

    return frequency, mask_xy, estimator_weights_xyz, map_xyz_mK, raw_hits_xyz


def effective_reconvolved_beam_sigma_deg(frequencies_mhz: np.ndarray, gamma: float) -> float:
    """Return the paper-style common effective Gaussian beam standard deviation.

    The native Gaussian beam is ``sigma = lambda / [sqrt(8 ln 2) D]``. The
    final effective beam is gamma times its widest (lowest-frequency) value,
    consistent with the paper's reconvolution description.
    """
    nu_low_hz = float(np.min(frequencies_mhz)) * 1e6
    sigma_rad = C_LIGHT_M_S / (np.sqrt(8.0 * np.log(2.0)) * nu_low_hz * DISH_DIAMETER_M)
    return float(gamma * np.rad2deg(sigma_rad))



def hitmap_thermal_noise(box, weights_xyz,mask,frequency,t_sample_s=2.0):
    """
    return thermal-noise cube for MeerKLASS    
    Parameters:
        box
        weights_xyz (array_like): 
            HITMAP .
        mask (array_like): 
            2d mask
        frequency (array_like)
            array of frequencies
        t_sample_s (float): 
            Observation time for each hit.
            
    Returns:
        noise_cube (array_like):
            noise cube in mK
    """
    
    hits_xyz = np.asarray(weights_xyz, dtype=float)
    mask_xy = np.asarray(mask, dtype=bool)
    
    
    valid_xyz = (hits_xyz > 0.0) & mask_xy[..., None]
    
    # The nominal MeerKLASS L-band channel spacing is 0.208984375 MHz.
    delta_nu_hz = float(np.mean(np.diff(frequency))) * 1e6    
    
    nu_ghz = frequency / 1000.0
    
    # Equations (21) and (22) of the paper. Temperatures are initially in K.
    T_rx_K = 7.5 + 10.0 * (nu_ghz - 0.75) ** 2
    T_gal_K = 25 * (408.0 / frequency) ** 2.75
    T_sys_K = T_rx_K + 3.0 + 2.725 + T_gal_K
    
    # Equation (20): sigma_N = Tsys / sqrt(2 * Delta_nu * t_p),
    # with t_p = N_hits * 2 s. Convert K to mK at the end.
    t_p_s = hits_xyz * t_sample_s
    sigma_thermal_mK = np.divide(
        1e3 * T_sys_K[None, None, :],
        np.sqrt(2.0 * delta_nu_hz * t_p_s),
        out=np.zeros_like(hits_xyz, dtype=float),
        where=valid_xyz,
    )
    
    # 1.0 reproduces the paper's analytic radiometer prediction.
    # Set this to 1.2 only when you deliberately want a phenomenological
    # match to the paper's median measured/theoretical RMS ratio, not a
    # replacement for real-map injection.
    noise_inflation = 1.0
    sigma_thermal_mK *= noise_inflation
    
    # A fresh seed / independent Generator must be used for every mock.
    rng = np.random.default_rng(42*2)
    thermal_noise_mK = rng.normal(loc=0.0, scale=sigma_thermal_mK)
    thermal_noise_mK[~valid_xyz] = 0.0

    valid_sigma_mK = sigma_thermal_mK[valid_xyz]
    return thermal_noise_mK
    
def pca_masked_weighted(nmodes, cube_mK, mask_xy, hits_xyz,  return_modes=False):
    """Remove foreground PCA modes from a masked native MeerKLASS cube.

    This follows the PCA convention in arXiv:2407.21626v2:
        C = (w o X) (w o X)^T / (n_theta - 1),
    where X has shape (Nnu, Ntheta) and w is constant along the
    frequency direction.  The foreground model is X_fg = A A^T X.

    Parameters
    ----------
    nmodes : int
        Number of largest-eigenvalue frequency modes to remove.
    cube_mK : ndarray, shape (Nx, Ny, Nnu)
        Observed/simulated intensity cube, in mK.
    mask_xy : ndarray[bool], shape (Nx, Ny)
        Common angular analysis mask.  True means retain the pixel.
    hits_xyz : ndarray, shape (Nx, Ny, Nnu)
        Raw two-second hit counts in the same native-grid ordering.
        It is used only to build the PCA covariance weight.
    return_modes : bool, default=False
        If True, also return the mixing matrix A and eigenvalues.

    Returns
    -------
    cleaned_mK : ndarray, shape (Nx, Ny, Nnu)
        PCA-cleaned field, with pixels outside the common usable sky set to 0.
    """
    cube_mK = np.asarray(cube_mK, dtype=float)
    mask_xy = np.asarray(mask_xy, dtype=bool)
    hits_xyz = np.asarray(hits_xyz, dtype=float)

    nx, ny, nnu = cube_mK.shape

    weights_xy = np.mean(hits_xyz, axis=-1)

    # the covariance or the foreground projection.
    finite_xy = np.all(np.isfinite(cube_mK), axis=-1)
    usable_xy = mask_xy & finite_xy & (weights_xy > 0.0)
    if usable_xy.sum() <= 1:
        raise ValueError("The common usable mask has fewer than two pixels.")

    w = weights_xy[usable_xy].astype(float)
    w /= np.mean(w)  # global rescaling leaves the PCA eigenvectors unchanged

    # X is frequency x angular-pixel. Only unmasked pixels enter PCA.
    X = cube_mK.reshape(nx * ny, nnu).T
    X_good = X[:, usable_xy.ravel()]

    # Equation (29): C = (w o X)(w o X)^T/(n_theta - 1).
    # Deliberately no subtraction of a per-frequency mean here.
    X_weighted = X_good * w[None, :]
    covariance = (X_weighted @ X_weighted.T) / (X_good.shape[1] - 1)
    covariance = 0.5 * (covariance + covariance.T)  # numerical symmetry

    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    A = eigvecs[:, order[:nmodes]]  # Nnu x nmodes mixing matrix

    cleaned = np.zeros_like(X)
    if nmodes == 0:
        cleaned[:, usable_xy.ravel()] = X_good
    else:
        X_fg_good = A @ (A.T @ X_good)
        cleaned[:, usable_xy.ravel()] = X_good - X_fg_good

    cleaned_mK = cleaned.T.reshape(nx, ny, nnu)

    if return_modes:
        return cleaned_mK, A, eigvals, usable_xy, weights_xy
    return cleaned_mK


