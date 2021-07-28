"""Void-finding and measurement routines."""

import numpy as np
import scipy.interpolate
from skimage.segmentation import watershed
import skimage.future.graph as graph
import time


def void_centroid(void_cat, void_labels, box, field=None, kind='uniform'):
    """
    Get the centroid coordinates of voids, using a choice of several methods.
    
    Parameters
    ----------
    void_cat : array_like
        Array of integer labels of void regions to calculate centroids for.
    
    void_labels : array_like of int
        Cube of integer labels for each voxel in the simulation box. Voxels 
        with the same label have been determined to belong to the same void. 
    
    box : CosmoBox object
        Object containing metadata about the simulation box.
    
    field : array_like, optional
        3D field used as weight for some kinds of centroid calculation method. 
        This should usually be the density contrast field, delta. Default: None.
    
    kind : str, optional
        What kind of centroid to calculate. Options are:
         - 'uniform': Uniform-weighted centroid (all voxels in region included).
         - 'minimum': Density minimum of each region. Set `field = delta`.
         - 'density': Density-weighted centroid (overdensities are ignored). 
                      Set `field = delta`.
    
    Returns
    -------
    centroids : array_like
        Array of shape (Nvoids, 3), containing the position vector (in box 
        comoving coordinates) of the centroid of each void.
    """
    void_labels_int = void_labels.astype(np.int)
    unique_lbls = void_cat.astype(np.int)
    
    # Loop over voids and find centroid
    centroids = {}
    for i, lbl in enumerate(unique_lbls):
        
        # Find voxels that belong to this region
        ix, iy, iz = idxs = np.where(void_labels_int == lbl)
        
        # If centroid kind defined as minimum point
        if kind == 'minimum':
            ii = np.argmin(field[idxs]) # index of minimum in idxs array
            centroids[lbl] = np.array([box.x[ix[ii]],
                                       box.y[iy[ii]], 
                                       box.z[iz[ii]]])
            continue
        
        # Determine weights (should be normalised to sum(w) = 1)
        if kind == 'uniform':
            w = 1. / ix.size
            
        elif kind == 'density':
            # Weights must be positive
            w = field[idxs].flatten()
            w *= -1. # Flip sign so underdensities have +ve weight
            w[w < 0.] = 0. # set overdensities to zero
            w /= np.sum(w)
        else:
            raise ValueError("Centroid kind '%s' not recognised." % weights)
        
        # Weighted centre of mass (just the mean of each component if weights are equal)
        centroids[lbl] = np.array([np.sum(w * box.x[ix]), 
                                   np.sum(w * box.y[iy]), 
                                   np.sum(w * box.z[iz])])
    return centroids    


def void_radii(void_cat, void_labels, box):
    """
    Calculate void radii, using a simple volume metric.
    
    Parameters
    ----------
    void_cat : array_like
        Array of integer labels of void regions to calculate radii for.
    
    void_labels : array_like of int
        Cube of integer labels for each voxel in the simulation box. Voxels 
        with the same label have been determined to belong to the same void. 
    
    box : CosmoBox object
        Object containing metadata about the simulation box.
    
    Returns
    -------
    radii : dict
        Dictionary of void IDs (key) and radii (value), for each void in 
        void_cat. The radii are in Mpc (same units as ``box.x``).
    """
    # Voxel volume
    dx = box.x[1] - box.x[0]
    dy = box.y[1] - box.y[0]
    dz = box.z[1] - box.z[0]
    dV = dx * dy * dz # comoving voxel size
    
    # Find radius of each void in the catalogue
    void_rad = {}
    for lbl in void_cat:
        ncells = np.where(void_labels == lbl)[0].size
        void_rad[lbl] = (3. * dV * ncells / (4.*np.pi))**(1./3.)
    return void_rad


def trim_by_volume(void_labels, nmin, nmax):
    """Remove labels for voids that have too few/too many voxels.
    
    Parameters
    ----------
    void_labels : array_like of int
        Cube of integer labels for each voxel in the simulation box. Voxels 
        with the same label have been determined to belong to the same void.
    
    nmin, nmax : int
        Minimum and maximum number of voxels allowed per void. Voids with a 
        number of voxels outside this range will be discarded.
    
    Returns
    -------
    void_cat : array_like
        Array of integer labels of void regions that pass the cut.
    """
    # Unique labels
    unique, counts = np.unique(void_labels, return_counts=True)
    
    # Trim labels that do not match the criteria
    return unique[np.logical_and(counts >= nmin, counts <= nmax)]


def apply_watershed(field, markers=None, mask_threshold=0., merge_threshold=0.2, 
                    verbose=True):
    """
    Apply a watershed algorithm to find voids in a given field.
    
    Candidate void regions are found using the `skimage.segmentation.watershed` 
    method. They are then merged together into a reduced set of voids using a 
    graph-based region merging methods, `skimage.future.graph.rag_mean_color` 
    and `cut_threshold`.
    
    Parameters
    ----------
    field : array_like
        3D field to apply the watershed algorithm to. This will be normalised 
        to produce a density contrast, so that `f = field / mean(field) - 1`.
    
    markers : int, optional
        Number of initial seeds of regions to place before running the 
        watershed algorithm. The seeds are placed in local minima of the field. 
        The final number of regions will be less than this number. 
        Default: None.
    
    mask_threshold : float, optional
        Mask (exclude) all regions with density contrast `f` above this value. 
        This is intended to exclude higher-density regions that are not allowed 
        to form part of a void. Default: 0.
    
    merge_threshold : float, optional
        Similarity threshold (in terms of mean density) to use to decide 
        whether to merge neighbouring regions. Larger values allow less similar 
        regions to merge. Default: 0.2.
    
    Returns
    -------
    region_lbls : array_like
        Array with same shape as `field` with integer label for each voxel. A 
        value of 0 denotes a voxel that is not part of any void.
    """
    # Normalise field to get density contrast
    if np.mean(field) == 0.:
        f = field / np.mean(field) - 1.
    else:
        f = field
    
    # Mask-out high-density regions
    mask = np.ones_like(f, dtype=np.bool)
    mask[np.where(f > mask_threshold)] = False
    
    # Apply watershed algorithm
    if verbose:
        print("Running watershed algorithm")
    t0 = time.time()
    region_lbls = watershed(f, markers=markers, mask=mask)
    if verbose:
        print("Watershed took %2.2f sec" % (time.time() - t0))
        print("No. regions:", np.unique(region_lbls).size)
    
    # Use a graph-based region merging algorithm
    t0 = time.time()
    if verbose:
        print("Running merging algorithm")
    g = graph.rag_mean_color(f, region_lbls, connectivity=1, sigma=2)
    region_lbls_new = graph.cut_threshold(region_lbls, g, merge_threshold)
    if verbose:
        print("Merging took %2.2f sec" % (time.time() - t0))
        print("No. regions after merging:", np.unique(region_lbls_new).size)
    
    return region_lbls_new


def stack_voids(void_cat, void_labels, box, field, centroid_kind='density', 
                grid_scale=1., grid_pix=31):
    """
    Stack the value of a field over a set of void regions. The voids are 
    centred and scaled by their radius before stacking, so the field values are 
    interpolated onto a standard grid before stacking.
    
    Masked arrays are used to compute the average value in each grid voxel. 
    Since the voids have irregular shapes, not all voids contribute to all grid 
    voxels, i.e. the number of values contributing to the average in each voxel 
    varies between voxels.
    
    Parameters
    ----------
    void_cat : array_like
        Array of integer labels of void regions to stack.
    
    void_labels : array_like of int
        Cube of integer labels for each voxel in the simulation box. Voxels 
        with the same label have been determined to belong to the same void. 
    
    box : CosmoBox object
        Object containing metadata about the simulation box.
    
    field : array_like
        3D field to be stacked. This should usually be the density contrast 
        field, delta.
    
    centroid_kind : str, optional
        Which type of centroid to calculate (see ``void_centroid()`` for 
        options). Default: 'density'.
    
    grid_scale : float, optional
        Bounds of the grid that the radius-normalised voids are interpolated 
        onto, given by [-grid_scale, +grid_scale]. Default: 1.
    
    grid_pix : int, optional
        Number of grid pixels in each dimension. Default: 31.
    
    Returns
    -------
    stacked_voids : array_like
        Grid of field values, averaged over all (stacked) voids.
    
    failures : list
        List of void IDs that could not be interpolated onto a grid.
    """
    # Compute void centroids
    centroids = void_centroid(void_cat=void_cat, void_labels=void_labels, 
                              box=box, field=field, kind='uniform')
    
    # Compute void radii
    radii = void_radii(void_cat=void_cat, void_labels=void_labels, box=box)
    
    # Define grid to interpolate onto
    grid = np.linspace(grid_scale, grid_scale, grid_pix)
    grid_x, grid_y, grid_z = np.meshgrid(grid, grid, grid)

    # Empty grid for summing
    void_grid_list = []

    # Loop over voids in void_cat
    failures = []
    for lbl in void_cat:
        
        # Get voxels belonging to this void
        idxs = np.where(void_labels == lbl)
        x_idx, y_idx, z_idx = idxs

        # Coordinates with respect to void centre, normalised by void radius
        _x = (box.x[x_idx] - centroids[lbl][0]) / radii[lbl]
        _y = (box.y[y_idx] - centroids[lbl][1]) / radii[lbl]
        _z = (box.z[z_idx] - centroids[lbl][2]) / radii[lbl]
        
        try:
            # Interpolate onto grid
            void_grid = scipy.interpolate.griddata(np.column_stack((_x,_y,_z)), 
                                                   field[idxs].flatten(), 
                                                   xi=(grid_x.flatten(), 
                                                       grid_y.flatten(), 
                                                       grid_z.flatten()), 
                                                   method='linear', 
                                                   fill_value=np.nan, 
                                                   rescale=False)
            void_grid = void_grid.reshape(grid_x.shape)
        except:
            failures.append(lbl)
            continue
        
        # Mask invalid values
        void_grid = np.ma.masked_invalid(void_grid)
        void_grid_list.append(void_grid)
    
    # Calculate the mean over all voids
    void_grid_arr = np.ma.array(void_grid_list)
    void_grid_mean = np.ma.mean(void_grid_arr, axis=0)
    return void_grid_mean, failures
    
