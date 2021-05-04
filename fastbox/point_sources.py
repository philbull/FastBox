"""
Make point source maps according to the Battye et al. 2012 recipe. Based on Eq 36 from https://arxiv.org/pdf/1209.0343.pdf
and https://www.research.manchester.ac.uk/portal/files/67403180/FULL_TEXT.PDF p99
"""
try:
    import healpy as hp
except:
    raise ImportError("Unable to import `healpy`; please install")
import numpy as np
import scipy.integrate as defint
import matplotlib.pyplot as plt

class PointSourceModel(object):

    def __init__(self, box):
        """
        An object to manage the point source model.

        Parameters
        ----------
        box : CosmoBox
            Object containing a simulation box.
        """

        self.box = box

        self.cst = {"kbolt": 1.3806488e-23, "light": 2.99792458e8, "plancks": 6.626e-34, "cmb_temp": 2.73}

    def battye(self, sjy):
        """empirical point source model of battye 2013 for intergrated flux """

        oooh = np.log10(sjy)

        sumbit = 2.593 * oooh**0 + 9.333 * 10**-2 * oooh**1. -4.839 * 10**-4 * oooh**2. \
                    + 2.488 * 10**-1 * oooh**3. + 8.995 * 10**-2 * oooh**4. + \
                        8.506 * 10**-3 * oooh**5.

        inte = (10.**sumbit) * sjy**(-2.5) * sjy

        return inte

    def pois(self, sjy):
        """empirical point source model of battye 2013 for poisson power spec """

        oooh = np.log10(sjy)

        sumbit = 2.593 * oooh**0 + 9.333 * 10**-2 * oooh**1. -4.839 * 10**-4 * oooh**2. \
                    + 2.488 * 10**-1 * oooh**3. + 8.995 * 10**-2 * oooh**4. + \
                        8.506 * 10**-3 * oooh**5.

        inte = (10.**sumbit) * sjy**(-2.5) * sjy**(2.0)

        return inte

    def numcount(self, sjy):
        """empirical point source model og battye 2013 for source count """

        oooh = np.log10(sjy)

        sumbit = 2.593 * oooh**0 + 9.333 * 10**-2 * oooh**1. -4.839 * 10**-4 * oooh**2. \
                    + 2.488 * 10**-1 * oooh**3. + 8.995 * 10**-2 * oooh**4. + \
                        8.506 * 10**-3 * oooh**5.

        inte = (10.**sumbit) * sjy**(-2.5)

        return inte

    def make_ps_nobeam(self, freqs, smax, beta, deltbeta):
        """ make a range of ps maps from nside, frequencies, cut-off flux in Jy and resolution in arcmin """

        # Angular pixels
        ang_x, ang_y = self.box.pixel_array(redshift=self.box.redshift)
        xside = len(ang_x)
        yside = len(ang_y)
        
        nside = 256
        ell = np.arange(nside*3) + 1.0
        npix = 12 * nside * nside
        pixarea = (np.degrees(4 * np.pi) * 60.) / (npix)
        lenf = len(freqs)
        cfact = self.cst["light"]**2 / (2 * self.cst["kbolt"] * (1.4e9)**2) * 10.**-26

        ######### first to make the point source map at 1.4 GHz ################
        # Get the mean temperature
        intvals = defint.quad(lambda sjy: self.battye(sjy), 0., smax)
        tps14 = cfact * (intvals[0] - intvals[1])

        #Get the clustering contribution
        clclust = 1.8 * 10**-4 * ell**-1.2 * tps14**2
        np.random.seed(0)
        clustmap = hp.sphtfunc.synfast(clclust, nside, new=True)

        #Get the poisson contribution
        #under 0.01 Jy poisson contributions behave as gaussians
        clpoislow = np.zeros((len(ell)))
        val = 0
        for ival in np.arange(1e-6, 0.01, (0.01-1e-6)/ len(ell)):
            intvals = defint.quad(lambda sjy: self.pois(sjy), 0., ival)
            clpoislow[val] = cfact**2 * (intvals[0] - intvals[1])
            val += 1
        np.random.seed(10)
        poislowmap = hp.sphtfunc.synfast(clpoislow, nside, new=True)

        shotmap = np.zeros((npix))
        #over 0.01 Jy you need to inject sources into the sky
        if smax > 0.01:
            for ival in np.arange(0.01, smax, (smax - 0.01)/10.):
                #N is number of sources per steradian per jansky
                numbster = defint.quad(lambda sjy: self.numcount(sjy), ival - 1e-3, ival + 1e-3)[0]
                numbsky = int(4 * np.pi * numbster * ival)
                tempval = cfact * defint.quad(lambda sjy: self.battye(sjy), 0.01, ival)[0] / pixarea
                randind = np.random.choice(range(npix), numbsky)
                shotmap[randind] = tempval

        map14 = tps14 + poislowmap + clustmap + shotmap

        #########################################################################
    
        alphabig = np.random.normal(beta, scale=deltbeta**2, size=npix)
    
        #### extract 256 by 256 square ########    
        rotation=(0., -62., 0.)
        reso_arcmin = hp.nside2resol(nside, arcmin=True)
    
        map14= hp.visufunc.gnomview(map14, coord='G', rot=rotation, xsize=xside, \
                    ysize=yside, reso=reso_arcmin, flip='astro', return_projected_map=True)
        plt.close()
        map14 = map14[::-1]
    
        alphas = hp.visufunc.gnomview(alphabig, coord='G', rot=rotation, xsize=xside, \
                    ysize=yside, reso=reso_arcmin, flip='astro', return_projected_map=True)
        plt.close()
        alphas = alphas[::-1]

        #######################################

        ######### scale up to different frequencies ################
        maps = np.zeros((xside, yside, lenf))
        for xxx in range(xside):
            for yyy in range(yside):
                maps[xxx, yyy, :] = np.array([map14[xxx, yyy] * (freqs[freval]/1400.)**(alphas[xxx, yyy]) for freval in range(lenf)])
        tps_mean = np.array([tps14 * (freqs[ival]/1400.)**(beta) for ival in range(lenf)]).reshape(lenf, 1)
        #########################################################################

        return maps*1e3, tps_mean*1e3
