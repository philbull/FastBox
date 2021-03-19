"""
Add dark matter halos to a density field, using Poisson statistics.
"""
import numpy as np
import pyccl as ccl
import pylab as plt


class HaloDistribution(object):

    def __init__(self, box, mass_range, mass_bins):
        """
        An object to manage the placing of halos on top of a realisation of a 
        density field in a box.
        
        Parameters
        ----------
        box : CosmoBox
            Object containing a simulation box.
        
        mass_range : tuple
            Range of halo masses, in Msun.
        
        mass_bins : int
            Number of mass bins.
        """
        self.box = box
        self.Mmin, self.Mmax = mass_range
        self.mass_bins = mass_bins
    
    
    def construct_bins(self, z):
        """
        Construct a binned halo mass function and bias function.
        """
        a = 1./(1.+z)
        
        # Define mass bins
        Mh_edges = np.logspace(np.log10(self.Mmin), 
                               np.log10(self.Mmax), 
                               int(self.mass_bins)+1)
        Mh_centres = 0.5 * (Mh_edges[1:] + Mh_edges[:-1])
        
        # Get mass function and bias at mass bin centres
        self.dndlog10M = ccl.massfunction.massfunc(self.box.cosmo, Mh_centres, 
                                              a, overdensity=200)
        self.bias = ccl.halo_bias(cosmo, Mh_centres, a, overdensity=200)
        
   
    def generate_halos(self, field_x, nbar):
        """
        Generate halo catalogue.
        """
        #self.box.x
        
        # Get volume of each voxel
        voxel_vol = self.box.Lx * self.box.Ly * self.box.Lz / self.box.N**3.
        
        Nbar = (1. + field_x) #voxel_vol * nbar * (1. + field_x)
        Nbar[np.where(Nbar < 0.)] = 0.
        Nbar *= voxel_vol * nbar
        
        print("Nbar:", np.min(Nbar), np.max(Nbar))
        
        Nhalo = np.random.poisson(lam=np.nan_to_num(Nbar))
        return Nhalo


