
import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d

LINE_FREQ = 1420.405752

def comoving_dimensions_from_survey(cosmo, angular_extent, freq_range=None, 
                                    z_range=None, line_freq=1420.405752):
    """
    Return the comoving dimensions and central redshift of a survey volume, 
    given its angular extent and redshift/frequency range.
    
    Parameters:
        cosmo (ccl.Cosmology object):
            CCL object that defines the cosmology to use.
        angular_extent (tuple):
            Angular extent of the survey in the x and y directions on the sky, 
            given in degrees. For example, `(10, 30)` denotes a 10 deg x 30 deg 
            survey area.
        freq_range (tuple, optional):
            The frequency range of the survey, given in MHz.
        z_range (tuple, optional):
            The redshift range of the survey.
        line_freq (float, optional):
            Frequency of the emission line used as redshift reference, in MHz.
        
    Returns:
        zc (float):
            Redshift of the centre of the comoving box.
        dimensions (tuple):
            Tuple of comoving dimensions of the survey volume, (Lx, Ly, Lz), 
            in Mpc.
    """
    # Check inputs
    if (freq_range is not None and z_range is not None) \
    or (freq_range is None and z_range is None):
        raise ValueError("Must specify either freq_range of z_range.")
    assert len(angular_extent) == 2, "angular_extent must be tuple of length 2"
    
    # Convert frequency range to z range if needed
    if freq_range is not None:
        assert len(freq_range) == 2, "freq_range must be tuple of length 2"
        z_range = (line_freq/freq_range[0] - 1., 
                   line_freq/freq_range[1] - 1.)
    assert len(z_range) == 2, "z_range must be tuple of length 2"
    
    # Sort z_range
    zmin, zmax = sorted(z_range)
    
    # Convert z_range to comoving r range
    rmin = ccl.comoving_radial_distance(cosmo, 1./(1.+zmin))
    rmax = ccl.comoving_radial_distance(cosmo, 1./(1.+zmax))
    Lz = rmax - rmin
    
    # Get comoving centroid of volume
    _z = np.linspace(zmin, zmax, 100)
    _r = ccl.comoving_radial_distance(cosmo, 1./(1.+_z))
    rc = 0.5 * (rmax + rmin)
    zc = interp1d(_r, _z, kind='linear')(rc)
    
    # Get transverse extent of box, evaluated at centroid redshift
    r_trans = ccl.comoving_angular_distance(cosmo, 1./(1.+zc))
    Lx = angular_extent[0] * np.pi/180. * r_trans
    Ly = angular_extent[1] * np.pi/180. * r_trans
    
    return zc, (Lx, Ly, Lz)

