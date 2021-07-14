"""Void-finding and measurement routines."""

import numpy as np
from skimage.segmentation import watershed
import skimage.future.graph as graph
import time


def void_centroids(void_labels, box, verbose=True):
    """Get the centroid coordinates of voids.
    
    Finds the mean position vector of the voxels that belong to each void, 
    which is an unweighted estimate of the centroid of the void.
    
    Parameters
    ----------
    void_labels : array_like of int
        Cube of integer labels for each voxel in the simulation box. Voxels 
        with the same label have been determined to belong to the same void. 
    
    box : CosmoBox object
        Object containing metadata about the simulation box.
    
    verbose : bool, optional
        Whether to print progress messages. Default: True.
    
    Returns
    -------
    centroids : array_like
        Array of shape (Nvoids, 3), containing the position vector (in box 
        comoving coordinates) of the centroid of each void.
    """
    void_labels_int = void_labels.astype(np.int)
    unique_lbls = np.unique(void_labels_int)
    
    # Loop over voids and find centroid
    centroids = []
    for i, lbl in enumerate(unique_lbls):
        if i % 1000 == 0 and verbose:
            print("%d / %d" % (i, unique_lbls.size))
        if lbl < 0:
            continue # inf or nan
        
        # Centre of mass (just the mean of each component if weights are equal)
        ix, iy, iz = np.where(void_labels_int == lbl)
        centroids.append([np.mean(ix), np.mean(iy), np.mean(iz)])
    
    # Rescale coordinates
    centroids = np.array(centroids)
    centroids[:,0] *= box.Lx / box.N
    centroids[:,1] *= box.Ly / box.N
    centroids[:,2] *= box.Lz / box.N
    return centroids
    

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
    f = field / np.mean(field) - 1.
    
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
    
