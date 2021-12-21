
import numpy as np
import scipy.interpolate


def replace_nan_with_channel_mean(field):
    """
    Replace NaN values in a datacube with the mean of the non-NaN values in 
    each channel (frequency slice).
    
    Parameters:
        field (array_like):
            3D datacube. The -1 axis will be assumed to be the frequency 
            direction.
    
    Returns:
        field_replaced (array_like):
            3D datacube with NaN values replaced.
    """
    # Work on a copy
    field_repl = field.copy().reshape((-1, field.shape[-1]))
    
    # Loop over frequency channels
    for j in range(field_repl.shape[-1]):
        idxs = np.isnan(field_repl[:,j]) # idxs of NaN values
        avg = np.mean(field_repl[~idxs,j])
        
        field_repl[idxs,j] = avg
    return field_repl.reshape(field.shape)


def interpolate_onto_grid(field, coords_orig, coords_new):
    """
    Interpolate a 3D field onto a grid with a different pixel/channel spacing.
    
    NOTE: The input set of coordinates should be in ascending order.
    
    Parameters:
        field (array_like):
            3D array of data.
        coords_orig (tuple):
            Tuple of coordinates of the original grid, in the order (x, y, z), 
            corresponding to field axes (0, 1, 2).
        coords_new (tuple):
            Tuple of coordinates of the new grid, in the order (x, y, z), 
            corresponding to field axes (0, 1, 2).
    
    Returns:
        field_new (array_like):
            3D array of data interpolated onto new grid shape.
    """
    # Extract original and new coordinates
    x, y, z = coords_orig
    x_new, y_new, z_new =coords_new
    
    # Replace NaN values in data
    field_nonan = replace_nan_with_channel_mean(field)

    # Construct interpolator for data
    # (Each array in coords_orig has to be ascending order)
    interp = scipy.interpolate.RegularGridInterpolator((x, y, z),
                                                       field_nonan, 
                                                       method='linear', 
                                                       bounds_error=False, 
                                                       fill_value=np.nan)

    # Build new grid with chosen resolution
    x3d, y3d, z3d = np.meshgrid(x_new, y_new, z_new)
    pts = np.array([x3d.flatten(), y3d.flatten(), z3d.flatten()])

    # Interpolate onto new fine grid
    field_new = interp(pts.T).reshape(x3d.shape)
    return field_new


def grid_catalogue(x, y, z, w=None, xlim=None, ylim=None, zlim=None, 
                   nx=None, ny=None, nz=None):
    """
    Bin a catalogue of 3D positions onto a regular grid.
    
    Parameters:
        x, y, z (array_like):
            3D positions of objects.
        w (array_like):
            Weight assigned to each object. Can be `None`.
        xlim, ylim, zlim (tuple):
            Tuples of the minimum and maximum coordinate values of the grid in 
            each dimension, e.g. `(xmin, xmax)`. If not specified, the ranges 
            of the data will be used.
        nx, ny, nz (int):
            Number of grid cells in each direction.
    """
    assert (nx is not None) and (ny is not None) and (nz is not None), \
        "nx, ny, and nz must be specified."
        
    # Get ranges
    if xlim is None:
        xlim = (np.min(x), np.max(x))
    if ylim is None:
        ylim = (np.min(y), np.max(y))
    if zlim is None:
        zlim = (np.min(z), np.max(z))
    xmin, xmax = xlim
    ymin, ymax = ylim
    zmin, zmax = zlim
    
    # Define grid
    xgrid = np.linspace(xmin, xmax, nx) # bin centres
    ygrid = np.linspace(ymin, ymax, ny)
    zgrid = np.linspace(zmin, zmax, nz)

    # Grid catalogue
    grid, _ = np.histogramdd(np.vstack([x, y, z]).T,
                                bins=(nx, ny, nz),
                                range=[(xmin, xmax), (ymin, ymax), (zmin, zmax)],
                                weights=w)
    return grid, (xgrid, ygrid, zgrid)
    
