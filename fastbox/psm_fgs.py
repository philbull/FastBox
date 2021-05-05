"""
Make foreground maps using the Planck FFP10 simulations
"""
try:
    import healpy as hp
except:
    raise ImportError("Unable to import `healpy`; please install")
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

class PSMfgModel(object):

    def __init__(self, box):
        """
        An object to manage the PSM foreground model.

        Parameters
        ----------
        box : CosmoBox
            Object containing a simulation box.
        """
        self.box = box

        self.cst = {"kbolt": 1.3806488e-23, "light": 2.99792458e8, "plancks": 6.626e-34, "cmb_temp": 2.73}

    def planckcorr(self, freq_ghz):
        """ Takes in frequency in GHZ and produces factor to be applied to temp """

        freq = freq_ghz * 10.**9.
        factor = self.cst["plancks"] * freq / (self.cst["kbolt"] * self.cst["cmb_temp"])
        correction = (np.exp(factor)-1.)**2. / (factor**2. * np.exp(factor))

        return correction

    def square_syncff(self, freq0, freeind):
        """
        determine the ff and synchrotron contributions for a given MHz frequency range

        OUTPUTS: sync amplitude in mK
                ff amplitude in mK
                sync spec ind        
        """

        # Angular pixels
        ang_x, ang_y = self.box.pixel_array(redshift=self.box.redshift)
        xside = len(ang_x)
        yside = len(ang_y)

        #admin
        s217loc = 'fastbox/COM_SimMap_synchrotron-ffp10-skyinbands-217_2048_R3.00_full.fits'
        f217loc = 'fastbox/COM_SimMap_freefree-ffp10-skyinbands-217_2048_R3.00_full.fits'
        s353loc = 'fastbox/COM_SimMap_synchrotron-ffp10-skyinbands-353_2048_R3.00_full.fits'

        #Convert maps from Tcmb to Trj
        sync217 = hp.fitsfunc.read_map(s217loc, field=0, nest=False) / self.planckcorr(217)
        sync353 = hp.fitsfunc.read_map(s353loc, field=0, nest=False) / self.planckcorr(353)
        free217 = hp.fitsfunc.read_map(f217loc, field=0, nest=False) / self.planckcorr(217)

        #get rid of unphysical values
        free217[np.where(free217 < 0.)[0]] = np.percentile(free217, 3)

        syncind = np.log(sync353/ sync217) / np.log(353./217.)

        synca = sync217 * ((freq0/1000.)/217.)**(syncind)
        freea = free217 * ((freq0/1000.)/217.)**(freeind)

        #need to fill in small scale structure for sync ind map which is MAMD's map at 5deg
        els = np.array(range(4000)) + 1.0
        cl5deg = hp.sphtfunc.anafast(np.random.normal(0.0, np.std(syncind), 12*2048*2048), lmax=4000)
        #power spectra taken from https://arxiv.org/pdf/astro-ph/0408515.pdf
        cells = cl5deg[0] * (1000./els)**(2.4)
        np.random.seed(90)
        syncind = syncind + hp.sphtfunc.synfast(cells, 2048)

        nside = hp.get_nside(syncind)      
        rotation=(0., -62., 0.)
        reso_arcmin = hp.nside2resol(nside, arcmin=True)
        nxpix = np.int(np.ceil(54.1*60./reso_arcmin))
        nypix = np.int(np.ceil(54.1*60./reso_arcmin))
    
        synca= hp.visufunc.gnomview(synca, coord='G', rot=rotation, xsize=nxpix, \
                    ysize=nypix, reso=reso_arcmin, flip='astro', return_projected_map=True)
        plt.close()
        synca = synca[::-1]
    
        zoom_param  = [xside, yside]/np.array(synca.shape)
    
        synca = ndimage.zoom(synca, zoom_param, order=3)
    
        freea = hp.visufunc.gnomview(freea, coord='G', rot=rotation, xsize=nxpix, \
                    ysize=nypix, reso=reso_arcmin, flip='astro', return_projected_map=True)
        plt.close()
        freea = freea[::-1]
        freea = ndimage.zoom(freea, zoom_param, order=3)
    
        syncind = hp.visufunc.gnomview(syncind, coord='G', rot=rotation, xsize=nxpix, \
                    ysize=nypix, reso=reso_arcmin, flip='astro', return_projected_map=True)
        plt.close()
        syncind = syncind[::-1]
        syncind = ndimage.zoom(syncind, zoom_param, order=3)

        return synca*1e3, freea*1e3, syncind
