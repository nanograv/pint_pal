## Tools/modules for running outlier analyses; will need to modify functions to accept model/toa objects from tc.

# Generic imports
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from astropy import log

# Epochalyptica imports
#import pint
#import pint.toa
import pint.fitter
#import pint.models.model_builder as mb
from pint.residuals import Residuals
import copy
from scipy.special import fdtr
from timing_analysis.utils import apply_cut_flag, apply_cut_select

"""
# Joanna's imports (should be able to clean this up quite a bit)
import matplotlib
from matplotlib.ticker import NullFormatter
import scipy.linalg as sl
import time
import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
#import libstempo as T (commented out, because ?!?!)
from enterprise.signals.selections import Selection
import scipy.linalg as sl, scipy.stats, scipy.special
import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
"""

def get_entPintPulsar(model,toas,sort=False):
    """Return enterprise.PintPulsar object with PINT model, toas embedded.

    Parameters
    ==========
    model: `pint.model.TimingModel` object
    toas: `pint.toa.TOAs` object

    Returns
    =======
    model: `enterprise.PintPulsar` object
        with drop_pintpsr=False (PINT model, toas embedded)
    """
    return PintPulsar(toas,model,sort=sort,planets=m.PLANET_SHAPIRO.value) # pickle kwarg?

def poutlier(p,likob):
    """Invoked on a sample parameter set and the appropriate likelihood,
    returns the outlier probability (a vector over the TOAs) and
    the individual sqrt(chisq) values"""
    
    # invoke the likelihood
    _, _ = likob.base_loglikelihood_grad(p)

    # get the piccard pulsar object
    # psr = likob.psr

    r = likob.detresiduals
    N = likob.Nvec

    Pb = likob.outlier_prob # a priori outlier probability for this sample
    P0 = likob.P0           # width of outlier range
    
    PA = 1.0 - Pb
    PB = Pb
    
    PtA = np.exp(-0.5*r**2/N) / np.sqrt(2*np.pi*N)
    PtB = 1.0/P0
    
    num = PtB * PB
    den = PtB * PB + PtA * PA
    
    return num/den, r/np.sqrt(N)

def get_outliers_piccard(epp,Nsamples=20000,Nburnin=1000):
    """Description

    Parameters
    ==========
    epp: `enterprise.PintPulsar` object

    Returns
    =======
    ???
    """
    # pta-outliers-specific imports
    import tempfile, pickle
    import scipy.linalg as sl, scipy.optimize as so
    import matplotlib.pyplot as plt
    import numdifftools as nd
    import corner
    from enterprise.pulsar import PintPulsar
    import interval as itvl
    from nutstrajectory import nuts6

    # Create interval likelihood object
    likob = itvl.Interval(epp)

    def func(x):
        ll, _ = likob.full_loglikelihood_grad(x)
        return -np.inf if np.isnan(ll) else ll

    def jac(x):
        _, j = likob.full_loglikelihood_grad(x)
        return j

    # Log likelihood for starting parameter vector
    ll_start = func(likob.pstart)

    endpfile = psr + '-endp.pickle'
    endp = likob.pstart
    for iter in range(3):
        res = so.minimize(lambda x: -func(x),
                          endp,
                          jac=lambda x: -jac(x),
                          hess=None,
                          method='L-BFGS-B', options={'disp': True})

        endp = res['x']
    pickle.dump(endp,open(endpfile,'wb'))    # Is this necessary?

    # Check func(endp) > ll_start?
    # To whiten the likelihood, start by computing the Hessian of the posterior. This takes some time.
    nhyperpars = likob.ptadict[likob.pname + '_outlierprob'] + 1
    hessfile = psr + '-fullhessian.pickle'
    reslice = np.arange(0,nhyperpars)

    def partfunc(x):
        p = np.copy(endp)
        p[reslice] = x
        return likob.full_loglikelihood_grad(p)[0]

    ndhessdiag = nd.Hessdiag(func)
    ndparthess = nd.Hessian(partfunc)

    # Create a good-enough approximation for the Hessian
    nhdiag = ndhessdiag(endp)
    nhpart = ndparthess(endp[reslice])
    fullhessian = np.diag(nhdiag)
    fullhessian[:nhyperpars,:nhyperpars] = nhpart
    pickle.dump(fullhessian,open(hessfile,'wb'))    # Is this necessary?

    # Whiten the likelihood object with Hessian in hand.
    wl = itvl.whitenedLikelihood(likob, endp, -fullhessian)

    # Sanity check (necessary? log.info/warning?)
    #likob.pstart = endp
    #wlps = wl.forward(endp)
    #print(likob.full_loglikelihood_grad(endp))
    #print(wl.likob.full_loglikelihood_grad(wl.backward(wlps)))

    # Time to sample...
    psr = epp.model.PSR.value
    chaindir = 'outlier_' + psr
    if not os.path.exists(chaindir):
        os.makedirs(chaindir)

    chainfile = chaindir + '/samples.txt'
    if not os.path.isfile(chainfile) or len(open(chainfile,'r').readlines()) < 19999:
        # Run NUTS for Nsamples, with a burn-in of Nburnin (target acceptance = 0.6)
        samples, lnprob, epsilon = nuts6(wl.loglikelihood_grad, Nsamples, Nburnin,
                                     wlps, 0.6,
                                     verbose=True,
                                     outFile=chainfile,
                                     pickleFile=chaindir + '/save')

    parsfile = psr + '-pars.npy'
    samples = np.loadtxt(chaindir + '/samples.txt')
    fullsamp = wl.backward(samples[:,:-2])
    funnelsamp = likob.backward(fullsamp)
    pars = likob.multi_full_backward(funnelsamp)
    np.save(parsfile,pars)

    # Make corner plot with posteriors of hyperparams (includes new outlier param)
    parnames = list(likob.ptadict.keys())
    corner.corner(pars[:,:nhyperpars],labels=parnames[:nhyperpars],show_titles=True);
    plt.savefig(psr + '-corner.pdf')

    pobsfile = psr + '-pobs.npy'
    nsamples = len(pars)
    nobs = len(likob.Nvec)

    # basic likelihood
    lo = likob

    outps = np.zeros((nsamples,nobs),'d')
    sigma = np.zeros((nsamples,nobs),'d')

    for i,p in enumerate(pars):
        outps[i,:], sigma[i,:] = poutlier(p,lo)

    out = np.zeros((nsamples,nobs,2),'d')
    out[:,:,0], out[:,:,1] = outps, sigma    
    np.save(pobsfile,out)

    avgps = np.mean(outps,axis=0)
    medps = np.median(outps,axis=0)
    spd = 86400.0   # seconds per day
    residualplot = psr + '-residuals.pdf'

    # Make a dead simple plot showing outliers
    outliers = medps > 0.1
    nout = np.sum(outliers)
    nbig = nout
    
    print("Big: {}".format(nbig))
    
    if nout == 0:
        outliers = medps > 5e-4
        nout = np.sum(outliers)
    
    print("Plotted: {}".format(nout))

    plt.figure(figsize=(15,6))

    psrobj = likob.psr

    # convert toas to mjds
    toas = psrobj.toas/spd

    # red noise at the starting fit point
    _, _ = likob.full_loglikelihood_grad(endp)
    rednoise = psrobj.residuals - likob.detresiduals

    # plot tim-file residuals (I think)
    plt.errorbar(toas,psrobj.residuals,yerr=psrobj.toaerrs,fmt='.',alpha=0.3)

    # red noise
    # plt.plot(toas,rednoise,'r-')

    # possible outliers
    plt.errorbar(toas[outliers],psrobj.residuals[outliers],yerr=psrobj.toaerrs[outliers],fmt='rx')

    plt.savefig(residualplot)

    # Want to apply -pout flags to toas object and write result (outlier.tim, or something?)

class Gibbs(object):
    def __init__(self, pta, model = 'mixture',tdf=4, m=0.01,
                 vary_df=True, theta_prior='beta', vary_alpha= True,
                 alpha=1e10, pspin=None):

        self.pta = pta

        # a-prior outlier probability
        self.mp = m ##confidence that the outlying proportion would be small
        self.theta_prior = theta_prior

        # spin period
        self.pspin = pspin

        # vary t-distribution d.o.f
        self.vary_df = vary_df

        # vary alpha
        self.vary_alpha = vary_alpha

        # For now assume one pulsar
        self._residuals = self.pta.get_residuals()[0]

        # which likelihood model
        self._lmodel = model

        # auxiliary variable stuff
        xs = [p.sample() for p in pta.params]
        self._b = np.zeros(self.pta.get_basis(xs)[0].shape[1])

        # for caching
        self.TNT = None
        self.d = None

        # outlier detection variables
        self._pout = np.zeros_like(self._residuals)
        self._z = np.zeros_like(self._residuals)
        if not vary_alpha:
            self._alpha = np.ones_like(self._residuals) * alpha
        else:
            self._alpha = np.ones_like(self._residuals)
        self._theta = self.mp
        self.tdf = tdf
        if model in ['t', 'mixture', 'vvh17']:
            self._z = np.ones_like(self._residuals)

    @property
    def params(self):
        ret = []
        for param in self.pta.params:
            ret.append(param)
        return ret

    def map_params(self, xs):
        return {par.name: x for par, x in zip(self.params, xs)}


    def get_hyper_param_indices(self):
        ind = []
        for ct, par in enumerate(self.params):
            if 'ecorr' in par.name or 'log10_A' in par.name or 'gamma' in par.name:
                ind.append(ct)
        return np.array(ind)


    def get_white_noise_indices(self):
        ind = []
        for ct, par in enumerate(self.params):
            if 'efac' in par.name or 'equad' in par.name:
                ind.append(ct)
        return np.array(ind)


    def update_hyper_params(self, xs):

        # get hyper parameter indices
        hind = self.get_hyper_param_indices()

        # get initial log-likelihood and log-prior
        lnlike0, lnprior0 = self.get_lnlikelihood(xs), self.get_lnprior(xs)
        xnew = xs.copy()
        for ii in range(10):

            # standard gaussian jump (this allows for different step sizes)
            q = xnew.copy()
            sigmas = 0.05 * len(hind)
            probs = [0.1, 0.15, 0.5, 0.15, 0.1]
            sizes = [0.1, 0.5, 1.0, 3.0, 10.0]
            scale = np.random.choice(sizes, p=probs)
            par = np.random.choice(hind, size=1) ####only one hyper param at a time
            q[par] += np.random.randn(len(q[par])) * sigmas * scale

            # get log-like and log prior at new position
            lnlike1, lnprior1 = self.get_lnlikelihood(q), self.get_lnprior(q)

            # metropolis step
            diff = (lnlike1 + lnprior1) - (lnlike0 + lnprior0)
            if diff > np.log(np.random.rand()):
                xnew = q
                lnlike0 = lnlike1
                lnprior0 = lnprior1
            else:
                xnew = xnew

        return xnew


    def update_white_params(self, xs):

        # get white noise parameter indices
        wind = self.get_white_noise_indices()

        xnew = xs.copy()
        lnlike0, lnprior0 = self.get_lnlikelihood_white(xnew), self.get_lnprior(xnew)
        for ii in range(20):

            # standard gaussian jump (this allows for different step sizes)
            q = xnew.copy()
            sigmas = 0.05 * len(wind)
            probs = [0.1, 0.15, 0.5, 0.15, 0.1]
            sizes = [0.1, 0.5, 1.0, 3.0, 10.0]
            scale = np.random.choice(sizes, p=probs)
            par = np.random.choice(wind, size=1)
            q[par] += np.random.randn(len(q[par])) * sigmas * scale

            # get log-like and log prior at new position
            lnlike1, lnprior1 = self.get_lnlikelihood_white(q), self.get_lnprior(q)

            # metropolis step
            diff = (lnlike1 + lnprior1) - (lnlike0 + lnprior0)
            if diff > np.log(np.random.rand()):
                xnew = q
                lnlike0 = lnlike1
                lnprior0 = lnprior1
            else:
                xnew = xnew
        return xnew

    def update_b(self, xs): 

        # map parameter vector
        params = self.map_params(xs)

        # start likelihood calculations
        loglike = 0

        # get auxiliaries
        Nvec = self._alpha**self._z * self.pta.get_ndiag(params)[0]
        phiinv = self.pta.get_phiinv(params, logdet=False)[0]
        residuals = self._residuals

        T = self.pta.get_basis(params)[0]
        if self.TNT is None and self.d is None:
            self.TNT = np.dot(T.T, T / Nvec[:,None])
            self.d = np.dot(T.T, residuals/Nvec)
        #d = self.pta.get_TNr(params)[0]
        #TNT = self.pta.get_TNT(params)[0]

        # Red noise piece
        Sigma = self.TNT + np.diag(phiinv)

        try:
            u, s, _ = sl.svd(Sigma)
            mn = np.dot(u, np.dot(u.T, self.d)/s)
            Li = u * np.sqrt(1/s)
        except np.linalg.LinAlgError:

            Q, R = sl.qr(Sigma)
            Sigi = sl.solve(R, Q.T)
            mn = np.dot(Sigi, self.d)
            u, s, _ = sl.svd(Sigi)
            Li = u * np.sqrt(1/s)

        b = mn + np.dot(Li, np.random.randn(Li.shape[0]))

        return b


    def update_theta(self, xs):

        if self._lmodel in ['t', 'gaussian']:
            return self._theta
        elif self._lmodel in ['mixture', 'vvh17']:
            n = len(self._residuals)
            if self.theta_prior == 'beta':
                mk = n * self.mp
                k1mm = n * (1-self.mp)
            else:
                mk, k1mm = 1.0, 1.0
            ret = scipy.stats.beta.rvs(np.sum(self._z) + mk,
                                       n - np.sum(self._z) + k1mm) ###from paper: k = sample size, m = 0.01
            return ret


    def update_z(self, xs):

        # map parameters
        params = self.map_params(xs)

        if self._lmodel in ['t', 'gaussian']:
            return self._z
        elif self._lmodel in ['mixture', 'vvh17']:
            Nvec0 = self.pta.get_ndiag(params)[0]
            Tmat = self.pta.get_basis(params)[0]

            Nvec = self._alpha * Nvec0
            theta_mean = np.dot(Tmat, self._b)
            top = self._theta * scipy.stats.norm.pdf(self._residuals,
                                                     loc=theta_mean,
                                                     scale=np.sqrt(Nvec))
            if self._lmodel == 'vvh17':
                top = self._theta / self.pspin

            bot = top + (1-self._theta) * scipy.stats.norm.pdf(self._residuals,
                                                               loc=theta_mean,
                                                               scale=np.sqrt(Nvec0))
            q = top / bot
            q[np.isnan(q)] = 1
            self._pout = q
    
            return scipy.stats.binom.rvs(1, list(map(lambda x: min(x, 1), q)))


    def update_alpha(self, xs): 

        # map parameters
        params = self.map_params(xs)

        if np.sum(self._z) >= 1 and self.vary_alpha:
            Nvec0 = self.pta.get_ndiag(params)[0]
            Tmat = self.pta.get_basis(params)[0]
            theta_mean = np.dot(Tmat, self._b)
            top = ((self._residuals-theta_mean)**2 * self._z / Nvec0 + self.tdf) / 2
            bot = scipy.stats.gamma.rvs((self._z+self.tdf)/2)
            return top / bot
        ##formular 12
        else:
            return self._alpha

    def update_df(self, xs):

        if self.vary_df:
            # 1. evaluate the log conditional posterior of df for 1, 2, ..., 30.
            log_den_df = np.array(map(self.get_lnlikelihood_df, np.arange(1,31)))

            # 2. normalize the probabilities
            den_df = np.exp(log_den_df - log_den_df.max())
            den_df /= den_df.sum()

            # 3. sample one of values (1, 2, ..., 30) according to the probabilities
            df = np.random.choice(np.arange(1, 31), p=den_df)

            return df
        else:
            return self.tdf


    def get_lnlikelihood_white(self, xs):

        # map parameters
        params = self.map_params(xs)
        matrix = self.pta.get_ndiag(params)[0]
        
        # Nvec and Tmat
        Nvec = self._alpha**self._z * matrix
        Tmat = self.pta.get_basis(params)[0]

        # whitened residuals
        mn = np.dot(Tmat, self._b)
        yred = self._residuals - mn

        # log determinant of N
        logdet_N = np.sum(np.log(Nvec))

        # triple product in likelihood function
        rNr = np.sum(yred**2/Nvec)

        # first component of likelihood function
        loglike = -0.5 * (logdet_N + rNr)

        return loglike


    # this can and should be much cleaner
    def get_lnlikelihood(self, xs):

        # map parameter vector
        params = self.map_params(xs)

        # start likelihood calculations
        loglike = 0

        # get auxiliaries
        Nvec = self._alpha**self._z * self.pta.get_ndiag(params)[0]
        phiinv, logdet_phi = self.pta.get_phiinv(params, logdet=True)[0]
        residuals = self._residuals

        T = self.pta.get_basis(params)[0]
        if self.TNT is None and self.d is None:
            self.TNT = np.dot(T.T, T / Nvec[:,None])
            self.d = np.dot(T.T, residuals/Nvec)
        #d = self.pta.get_TNr(params)[0]
        #TNT = self.pta.get_TNT(params)[0]

        # log determinant of N
        logdet_N = np.sum(np.log(Nvec))

        # triple product in likelihood function
        rNr = np.sum(residuals**2/Nvec)

        # first component of likelihood function
        loglike += -0.5 * (logdet_N + rNr)

        # Red noise piece
        Sigma = self.TNT + np.diag(phiinv)

        try:
            cf = sl.cho_factor(Sigma)
            expval = sl.cho_solve(cf, self.d)
        except np.linalg.LinAlgError:
            return -np.inf

        logdet_sigma = np.sum(2 * np.log(np.diag(cf[0])))
        loglike += 0.5 * (np.dot(self.d, expval) - logdet_sigma - logdet_phi)

        return loglike

    def get_lnlikelihood_df(self, df):
        n = len(self._residuals)
        ll = -(df/2) * np.sum(np.log(self._alpha)+1/self._alpha) + \
            n * (df/2) * np.log(df/2) - n*scipy.special.gammaln(df/2)
        return ll

    def get_lnprior(self, xs): ###use np.sum(np.fromiter(generator))

        return np.sum(p.get_logpdf(x) for p, x in zip(self.params, xs))


    def sample(self, xs, niter=10000):

        self.chain = np.zeros((niter, len(xs)))
        self.bchain = np.zeros((niter, len(self._b)))
        self.thetachain = np.zeros(niter)
        self.zchain = np.zeros((niter, len(self._residuals)))
        self.alphachain = np.zeros((niter, len(self._residuals)))
        self.poutchain = np.zeros((niter, len(self._residuals)))
        self.dfchain = np.zeros(niter)

        xnew = xs
        tstart = time.time()
        outdir = 'outlier_J1909/'
        os.system('mkdir -p {}'.format(outdir))
        for ii in range(niter):
            self.chain[ii, :] = xnew
            self.bchain[ii,:] = self._b
            self.zchain[ii,:] = self._z
            self.thetachain[ii] = self._theta
            self.alphachain[ii,:] = self._alpha
            self.dfchain[ii] = self.tdf
            self.poutchain[ii, :] = self._pout

            self.TNT = None
            self.d = None

            # update white parameters
            xnew = self.update_white_params(xnew)

            # update hyper-parameters
            xnew = self.update_hyper_params(xnew)

            # if accepted update quadratic params
            if np.all(xnew != self.chain[ii,-1]):
                self._b = self.update_b(xnew)

            # update outlier model params
            self._theta = self.update_theta(xnew)
            self._z = self.update_z(xnew)
            self._alpha = self.update_alpha(xnew)
            self.tdf = self.update_df(xnew)

            if ii % 100 == 0 and ii > 0:
                sys.stdout.write('\r')
                sys.stdout.write('Finished %g percent in %g seconds.'%(ii / niter * 100, time.time()-tstart))
                sys.stdout.flush()
                
        np.savetxt('{}/chain.txt'.format(outdir), gibbs.chain)
        np.savetxt('{}/bchain.txt'.format(outdir), gibbs.bchain)
        np.savetxt('{}/zchain.txt'.format(outdir), gibbs.zchain)
        np.savetxt('{}/poutchain.txt'.format(outdir), gibbs.poutchain)
        np.savetxt('{}/thetachain.txt'.format(outdir), gibbs.thetachain)
        np.savetxt('{}/alphachain.txt'.format(outdir),gibbs.alphachain)

def Ftest(chi2_1, dof_1, chi2_2, dof_2):
    """
    Ftest(chi2_1, dof_1, chi2_2, dof_2):
        Compute an F-test to see if a model with extra parameters is
        significant compared to a simpler model.  The input values are the
        (non-reduced) chi^2 values and the numbers of DOF for '1' the
        original model and '2' for the new model (with more fit params).
        The probability is computed exactly like Sherpa's F-test routine
        (in Ciao) and is also described in the Wikipedia article on the
        F-test:  http://en.wikipedia.org/wiki/F-test
        The returned value is the probability that the improvement in
        chi2 is due to chance (i.e. a low probability means that the
        new fit is quantitatively better, while a value near 1 means
        that the new model should likely be rejected).
        If the new model has a higher chi^2 than the original model,
        returns value of False
    """
    delta_chi2 = chi2_1 - chi2_2
    if delta_chi2 > 0:
      delta_dof = dof_1 - dof_2
      new_redchi2 = chi2_2 / dof_2
      F = (delta_chi2 / delta_dof) / new_redchi2
      ft = 1.0 - fdtr(delta_dof, dof_2, F)
    else:
      ft = False
    return ft

def epochalyptica(model,toas,outfile='out.txt',ftest_threshold=1.0e-6):
    """ This does things

    Parameters:
    ===========

    Returns:
    ========

    """
    f = pint.fitter.GLSFitter(toas,model)
    chi2_init = f.fit_toas()
    ndof_init = pint.residuals.Residuals(toas,model).dof
    ntoas_init = toas.ntoas
    redchi2_init = chi2_init / ndof_init

    filenames = toas.get_flag_value('name')[0]
    fout = open(outfile,'w')
    numepochs = len(set(filenames))
    log.info(f'There are {numepochs} epochs (filenames) to analyze.')
    snrs = toas.get_flag_value('snr')[0]
    for filename in set(filenames):
        maskarray = np.ones(len(filenames),dtype=bool)
        receiver = None
        mjd = None
        toaval = None
        dmxindex = None
        dmxlower = None
        dmxupper = None
        sum = 0.0
        # Note, t[1]: mjd, t[2]: mjd (d), t[3]: error (us), t[6]: flags dict
        for index,t in enumerate(toas.table):
            if t[6]['name'] == filename:
                if receiver == None:
                    receiver = t[6]['f']
                if mjd == None:
                    mjd = int(t[1].value)
                if toaval == None:
                    toaval = t[2]
                    i = 1
                    while dmxindex == None:
                        DMXval = f"DMXR1_{i:04d}"
                        lowerbound = getattr(model.components['DispersionDMX'],DMXval).value
                        DMXval = f"DMXR2_{i:04d}"
                        upperbound = getattr(model.components['DispersionDMX'],DMXval).value
                        if toaval > lowerbound and toaval < upperbound:
                            dmxindex = f"{i:04d}"
                            dmxlower = lowerbound
                            dmxupper = upperbound
                        i += 1
                sum = sum + 1.0 / (float(t[3])**2.0)
                maskarray[index] = False
    
        toas.select(maskarray)
        f.reset_model()
        numtoas_in_dmxrange = 0
        for toa in toas.table:
            if toa[2] > dmxlower and toa[2] < dmxupper:
                numtoas_in_dmxrange += 1
        newmodel = model
        if numtoas_in_dmxrange == 0:
            log.debug(f"Removing DMX range {dmxindex}")
            newmodel = copy.deepcopy(model)
            newmodel.components['DispersionDMX'].remove_param(f'DMXR1_{dmxindex}')
            newmodel.components['DispersionDMX'].remove_param(f'DMXR2_{dmxindex}')
            newmodel.components['DispersionDMX'].remove_param(f'DMX_{dmxindex}')
        f = pint.fitter.GLSFitter(toas,newmodel)
        chi2 = f.fit_toas()
        ndof = pint.residuals.Residuals(toas,newmodel).dof
        ntoas = toas.ntoas
        redchi2 = chi2 / ndof
        if ndof_init != ndof:
            ftest = Ftest(float(chi2_init),int(ndof_init),float(chi2),int(ndof))
        else:
            ftest = False
        fout.write(f"{filename} {receiver} {mjd:d} {(ntoas_init - ntoas):d} {ftest:e} {1.0/np.sqrt(sum)}\n")
        toas.unselect()
    fout.close()

    ftest_results = np.genfromtxt(outfile,dtype=None,encoding=None)
    epochs_to_drop = []
    for line in ftest_results:
        if line[4] < ftest_threshold:
            epochs_to_drop.append(line[0])

    # Make the cuts.
    toas_to_cut = [(f in epochs_to_drop) for f in filenames]
    apply_cut_flag(toas,toas_to_cut,'epochdrop',warn=True)  # Useful to see if overlap with bad-toa
    apply_cut_select(toas,cut_flag_values=['epochdrop'],reason='epoch drop analysis')
