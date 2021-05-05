"""
Perform a least squares fit per pixel
"""

from lmfit import Minimizer, Parameters
import numpy as np
from multiprocessing import Queue, Process
import scipy.linalg as scl
from fastbox.point_sources import PointSourceModel
from fastbox.psm_fgs import PSMfgModel

class LSQfitting(object):

    def __init__(self, box):
        """
        An object to perform LSQ fit.

        Parameters
        ----------
        box : CosmoBox
            Object containing a simulation box.
        """
        self.box = box

    def resid_sy(self, params, freqs, data, **kwargs):
        """ Synchrotron model residuals """
    
        freqS = kwargs['freqS']
        noise = kwargs['noise']
    
        betaS = params['betaS']
        ampS = params['ampS']
    
        x_ghz = np.array(freqs)

        tot = ampS * (x_ghz / freqS) ** (betaS)

        weights = 1./noise**2

        return weights * (tot - data)

    def do_loop(self, int, bits, data, noise, freqs, bsval, syamp, ffamp, mod, bidea, freeind, queue):

        lenf = len(freqs)
        star = bits[int]
        enl = bits[int+1]

        for xno in range(star, enl):

            tval = data[xno, :]
            noval = noise[xno, :]
            bgu = bidea[xno]
        
            kwsdict = {'noise':noval, 'freqS':freqs[0]}
            params = Parameters()
            params.add('betaS', value=bgu, min=bgu*1.1, max=bgu*0.9)
            params.add('ampS', value=tval[0]*0.9, min=tval[0]*0.5, max=tval[0]*1.5)
 
            resultpre = Minimizer(self.resid_sy, params, fcn_args=(freqs, tval), fcn_kws=kwsdict)
            result = resultpre.minimize('least_sqaures')
            val_dic = result.params

            bsval[xno] = np.array(val_dic["betaS"])
    
            #getting amps from fitted specs
            specs = np.zeros((lenf, 2))
            specs[:, 0] = (freqs / freqs[0]) ** np.array(val_dic["betaS"])
            specs[:, 1] = (freqs / freqs[0]) ** np.array(freeind)
    
            num = np.dot(specs.T, tval)
            denom = scl.inv(np.dot(specs.T, specs))
            amps = np.dot(num, denom)
        
            syamp[xno] = amps[0]
            ffamp[xno] = amps[1]
        
            mod[xno,:] = np.dot(amps, specs.T)

        queue.put([bsval[star:enl], syamp[star:enl], ffamp[star:enl], mod[star:enl, :], star, enl])
    
    def run_fit(self, maps, freqs, numpix, tpsmean, freeind):
        """ perform the fit to data """

        lenf = len(freqs)
    
        #noise maps, my noise is at the level of free-free emission
        psmmodel = PSMfgModel(self.box) 
        _, freen, _ = psmmodel.square_syncff(900., freeind)
        sigma = np.std(freen)
        sigmas = sigma * (freqs/900.)**(freeind)
        noise = np.array([np.random.normal(loc=0.0, scale=sigmas[ii], size=numpix) for ii in range(lenf)])

        data = maps.reshape(numpix, lenf)- tpsmean.reshape(lenf, 1).T

        bsval = np.zeros((numpix))
        syamp = np.zeros((numpix))
        ffamp = np.zeros((numpix))
        mod = np.zeros(( numpix, lenf))

        bput = np.log(data[:,3] / data[:,0]) / np.log(freqs[3] / freqs[0])

        queue = Queue()
        bits = np.linspace(0, numpix, 8).astype('int')
        processes = [Process(target=self.do_loop, args=(intv, bits, data, noise.T, freqs, bsval, \
                        syamp, ffamp, mod, bput, freeind, queue)) for intv in range(8-1)]
 
        for p in processes:
            p.start()
        for p in processes:
            result = queue.get()
            syamp[result[4]:result[5]] = result[1]
            bsval[result[4]:result[5]] = result[0]
            ffamp[result[4]:result[5]] = result[2]
            mod[result[4]:result[5], :] = result[3]
        for p in processes:
            p.join()

        del queue, p, result

        return data - mod, bsval

    def give_hest(self, freqs, T_obs, freeind, psaveind, cuttoff, indspread):

        print('running LSQ fit')

        ang_x, ang_y = self.box.pixel_array(redshift=self.box.redshift)
        xside = len(ang_x)
        yside = len(ang_y)

        psmodel = PointSourceModel(self.box)
        _, tpsmean = psmodel.make_ps_nobeam(freqs, cuttoff, psaveind, freeind)

        ress, spec = self.run_fit(T_obs, freqs, xside*yside, tpsmean, freeind)
        n_ch = len(freqs)
        resid = ress.reshape(n_ch, xside, yside)
        bspec = spec.reshape(xside, yside)

        return resid, bspec






