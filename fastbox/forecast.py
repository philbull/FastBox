import numpy as np
import pyccl as ccl

C = 299792.458  # Speed of light, km/s
NU21CM = 1420.405751  # MHz
INF_NOISE = 1e50  # np.inf #1e100


# Example experiment definitions

# MeerKAT single-dish, 64 dishes, UHF band
# 580 - 1015 MHz
inst_meerkatuhf = {
    "name": "MeerKAT_UHF",
    "type": "dish",
    "D": 13.5,
    "Ndish": 64,
    "fsky": 0.1,
    "Tsys": 26.0,  # in K
    "ttot": 4000.0,  # hrs
    "fsky_overlap": 0.1,
    "kmax0": 0.14,  # Mpc^-1
}

# GBT single-dish, 7-beam receiver
inst_gbt = {
    "name": "GBT",
    "type": "dish",
    "D": 100.0,
    "Ndish": 7,
    "fsky": 0.15,
    "Tsys": 30.0,  # in K
    "ttot": 3.2e4,  # hrs
    "fsky_overlap": 0.15,
    "kmax0": 0.14,  # Mpc^-1
}

# Example HIRAX interferometer
inst_hirax = {
    "name": "hrx",
    "type": "interferometer",
    "D": 6.0,  # m
    "d_min": 6.0,  # m
    "d_max": 32.0 * 6.0 * 1.41,  # m
    "Ndish": 32 * 32,
    "fsky": 0.4,
    "Tsys": 50.0,  # in K
    "ttot": 2.8e4,  # hrs
    "fsky_overlap": 0.4,
    "kmax0": 0.14,  # Mpc^-1
}


def sigmaT(expt):
    """
    Calculate noise RMS, sigma_T, for an instrumental setup. In mK.MHz.

    Parameters:
        expt (dict):
            Dictionary of instrument parameters. See `fastbox.forecast.inst_gbt`
            for an example.

    Returns:
        sigmaT (float):
            Noise rms in mK.MHz.
    """
    sigmaT2 = (
        4.0
        * np.pi
        * expt["fsky"]
        * expt["Tsys"] ** 2
        / (expt["ttot"] * 3600.0 * expt["Ndish"])
    )
    return np.sqrt(sigmaT2)


def Tb(z):
    """
    Brightness temperature Tb(z), in mK. Uses a simple power-law fit to Mario
    Santos' model (powerlaw M_HI function with alpha=0.6).

    Parameters:
        z (float):
            Redshift.

    Returns:
        Tb (float):
            Brightness temperature, in mK.
    """
    return 5.5919e-02 + 2.3242e-01 * z - 2.4136e-02 * z ** 2.0


def bias_HI(z):
    """
    HI bias as a function of redshift, obtained using a simple polynomial fit
    to Mario Santos' model.

    Parameters:
        z (float):
            Redshift.

    Returns:
        bias (float):
            HI bias.
    """
    return 6.6655e-01 + 1.7765e-01 * z + 5.0223e-02 * z ** 2.0


def bias_gal(z):
    """
    Galaxy bias as a function of redshift, assuming a very simple "ELG-like"
    model, which is actually just b = sqrt(1 + z).

    Parameters:
        z (float):
            Redshift.

    Returns:
        bias (float):
            Galaxy bias.
    """
    return np.sqrt(1.0 + z)


def lmax_for_redshift(cosmo, z, kmax0=0.2):
    """
    Calculates an lmax for a bin at a given redshift. This is found by taking
    some k_max at z=0, scaling it by the growth factor, and converting to an
    ell value.

    Parameters:
        cosmo (ccl.Cosmology):
            CCL Cosmology object.
        z (float):
            Redshift.
        kmax (float):
            Max. wavenumber (cutoff) at z=0, in Mpc^-1 units.

    Returns:
        lmax (float):
            Maximum ell for this redshift.
    """
    r = ccl.comoving_radial_distance(cosmo, 1.0 / (1.0 + z))
    D = ccl.growth_factor(cosmo, 1.0 / (1.0 + z))
    lmax = r * D * kmax0
    return lmax


def lmin_for_redshift(cosmo, z, dmin):
    """
    Calculates lmin for an interferometer with a given minimum baseline length
    (dmin, in m).1

    Parameters:
        cosmo (ccl.Cosmology):
            CCL Cosmology object.
        z (float):
            Redshift.
        dmin (float):
            Minimum baseline length, in metres.

    Returns:
        lmin (float):
            Minimum ell for this redshift.
    """
    nu = 1420.0 / (1.0 + z)
    lam = (C * 1e3) / (nu * 1e6)  # wavelength in m
    lmin = 2.0 * np.pi * dmin / lam
    return lmin


def noise_im(cosmo, expt, ells, zmin, zmax, kmax_cutoff=False):
    """
    Noise (angular) power spectrum for a 21cm IM experiment, using expressions
    from Alonso et al. (arXiv:1704.01941). The noise expression used depends on
    the experiment type, either `expt['type'] == 'interferometer'` or `'dish'`.

    Parameters:
        cosmo (ccl.Cosmology):
            CCL Cosmology object.
        expt (dict):
            Dictionary of instrument parameters. See `fastbox.forecast.inst_gbt`
            for an example. The `'type'` key determines which noise expression
            is used.
        ells (array_like):
            ell values to calculate the noise power spectrum at.
        zmin, zmax (float):
            Minimum and maximum redshift of the redshift bin.
        kmax_cutoff (bool):
            Whether to apply a hard kmax cutoff (in ell-space) ornot.

    Returns:
        N_ell (array_like):
            Noise angular power spectrum as a function of ell, in units of mK^2.
    """
    ells = np.atleast_1d(ells)
    zmin = np.atleast_1d(zmin)
    zmax = np.atleast_1d(zmax)

    # Frequency scaling
    zc = 0.5 * (zmin + zmax)
    nu = NU21CM / (1.0 + zc)
    lam = (C * 1e3) / (nu * 1e6)  # wavelength in m

    # Bin widths and grid in SH wavenumber / wavelength
    dnu = NU21CM * (1.0 / (1.0 + zmin) - 1.0 / (1.0 + zmax))
    _ell, _lam = np.meshgrid(ells, lam)

    # Use approximate interferometer noise expression, from Eq. 8 of Alonso.
    if expt["type"] == "interferometer":
        # Angular scaling
        f_ell = np.exp(
            _ell
            * (_ell + 1.0)
            * (1.22 * _lam / expt["d_max"]) ** 2.0
            / (8.0 * np.log(2.0))
        )

        # Construct noise covariance
        N_ij = f_ell * sigmaT(expt) ** 2.0 / dnu[:, None]
        # FIXME: Is this definitely channel bandwidth, rather than total bandwidth?

        # Apply large-scale cut
        N_ij[np.where(_ell * _lam / (2.0 * np.pi) <= expt["d_min"])] = INF_NOISE

    elif expt["type"] == "dish":
        # Single-dish experiment noise expression
        # (Ndish already included in sigma_T expression)
        fwhm = 1.22 * _lam / expt["D"]
        B_l = np.exp(-_ell * (_ell + 1) * fwhm ** 2.0 / (16.0 * np.log(2.0)))
        N_ij = sigmaT(expt) ** 2.0 / dnu[:, None] / B_l ** 2.0

    else:
        raise NotImplementedError("Unrecognised instrument type '%s'." % expt["type"])

    # Transpose to get correct shape
    N_ij = N_ij.T

    # Apply kmax cutoff
    if kmax_cutoff:
        lmax = lmax_for_redshift(cosmo, zmax, kmax0=expt["kmax0"])
        for i in range(N_ij.shape[1]):
            idx = np.where(ells > lmax[i])
            N_ij[idx, i] = INF_NOISE

    # Add infinite noise outside experiment bandwidth (TODO)

    return N_ij


def number_density_to_area_density(cosmo, ngal, zmin, zmax, degrees=False):
    """
    Convert comoving galaxy number density (in Mpc^-3) to an area density (per
    unit solid angle).

    Parameters:
        cosmo (ccl.Cosmology):
            CCL Cosmology object.
        ngal (float):
            Comoving galaxynumber density, in Mpc^-3.
        zmin, zmax (float):
            Minimum and maximum redshift of the redshift bin.
        degrees (bool):
            Whether to return the area density in degrees^2 or steradians.

    Returns:
        Ngal (float):
            Galaxy number density in this redshift bin, per unit solid angle.
    """
    # Calculate comoving distances
    rmin = ccl.comoving_radial_distance(cosmo, a=1.0 / (1.0 + zmin))
    rmax = ccl.comoving_radial_distance(cosmo, a=1.0 / (1.0 + zmax))

    # Calculate volume of shell, in Mpc^3
    vol = (4.0 / 3.0) * np.pi * (rmax ** 3.0 - rmin ** 3.0)

    # Calculate total no. of galaxies in shell and divide by area of sky
    Ngal = (ngal * vol) / (4.0 * np.pi)  # No. gals. per steradian
    if degrees:
        return Ngal * (np.pi / 180.0) ** 2.0  # No. gals per deg^2
    else:
        return Ngal  # No. gals per ster.


def tracer_spectro(cosmo, zmin, zmax, kind="galaxy"):
    """
    Create a spectroscopic CCL tracer object with the right bias and selection
    function, for either a galaxy survey or an IM survey.

    Parameters:
        cosmo (ccl.Cosmology):
            CCL Cosmology object.
        zmin, zmax (float):
            Minimum and maximum redshift of the spectroscopic redshift bin.
        kind (str):
            The kind of tracer (can be 'galaxy' or 'im').

    Returns:
        tracer (ccl.NumberCountsTracer):
            A CCL tracer object with the right bias/selection function.
    """
    # Number counts/selection function in this tomographic redshift bin
    z = np.linspace(zmin * 0.8, zmax * 1.2, 2000)  # Pad zmin, zmax slightly
    tomo = np.zeros(z.size)
    tomo[np.where(np.logical_and(z >= zmin, z < zmax))] = 1.0

    # Define bias factors/selection function
    if kind == "galaxy":
        bz = bias_gal(z)
    else:
        # Clustering bias and 21cm monopole temperature, in mK
        bz = bias_HI(z) * Tb(z)

    # Number density tracer object
    n_spectro = ccl.NumberCountsTracer(
        cosmo, has_rsd=False, mag_bias=None, dndz=(z, tomo), bias=(z, bz)
    )
    return n_spectro


def fisher_bandpowers(
    ells, delta_ell, fsky, Cell_gal, Cell_im, Cell_cross, Nell_gal, Nell_im
):
    """
    Use a simplified version of the Fisher matrix from Eq. 24 of Padmanabhan
    et al. (arXiv:1909.11104) for the cross-spectrum bandpowers.

    Parameters:
        ells (array_like):
            Array of ell bin centres.
        delta_ell (float):
            ell bin width.
        fsky (float):
            Fraction of the sky covered by the survey.
        Cell_gal, Cell_im, Cell_cross (array_like):
            Angular power spectra for the galaxy survey, the IM survey, and
            their cross-spectrum, for a single z bin, as a function of ell.
        Nell_gal, Nell_im (array_like):
            Noise angular power spectra for the galaxy survey and IM survey.

    Returns:
        F_ell (array_like):
            Diagonal of the Fisher matrix as a function of ell. There are no
            off-diagonal terms in the Fisher matrix in this model. The units
            are (mK)^-2, i.e. the inverse of the units of the cross-spectrum
            squared.
    """
    # Calculate numerator (survey vol. factor etc.)
    numerator = (2.0 * ells + 1.0) * delta_ell * fsky

    # Calculate denominator (product of variances)
    denom = (Cell_gal + Nell_gal) * (
        Cell_im + Nell_im
    ) + Cell_cross ** 2.0  # units of (mK)^2

    return numerator / denom
