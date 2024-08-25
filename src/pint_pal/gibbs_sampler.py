import numpy as np
from tqdm import tqdm
import scipy.linalg as sl
from functools import cached_property
import os
import glob
import warnings
from enterprise_extensions import model_utils, blocks
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise.signals import signal_base, gp_signals
from scipy.linalg import solve_triangular as st_solve
from scipy.linalg import cho_factor, cho_solve


class GibbsSampler(object):

    """
    The Gibbs Method class used for single-pulsar noise analyses.

    Based on:

        Article by van Haasteren & Vallisneri (2014),
        "New advances in the Gaussian-process approach
        to pulsar-timing data analysis",
        Physical Review D, Volume 90, Issue 10, id.104012
        arXiv:1407.1838

        Initial structure of the code is based on https://github.com/jellis18/gibbs_student_t

    Authors:

        S. R. Taylor
        N. Laal
        J. G. Baier
    """

    def __init__(
        self,
        psr=None,
        Tspan=None,
        select="backend",
        vary_wn=False,
        inc_ecorr=False,
        ecorr_type="kernel",
        noise_dict=None,
        tm_marg=False,
        vary_rn=True,
        rn_components=30,
        tnequad=True,
        log10rhomin=-9.0,
        log10rhomax=-4.0,
        vary_dm=False,
        dm_components=50,
        vary_chrom=False,
        chrom_components=50,
        include_quadratic=False,
    ):
        """
        Parameters
        -----------

        psr : object
            instance of an ENTERPRISE psr object for a single pulsar

        Tspan: float (optional)
            if given, the baseline of the pulsar is fixed to the input value. If not,
            baseline is determined inetrnally

        select: str
            the selection of backend ('backend' or 'none') for the white-noise parameters

        self.vary_wn: bool
            whether to vary the white noise

        inc_ecorr: bool
            whether to include ecorr

        ecorr_type: str
            the type of ecorr to use. Choose between 'basis' or 'kernel'

        noise_dict: dict
            white noise dictionary in case 'self.vary_wn' is set to False

        tm_marg: bool
            whether to marginalize over timing model parameters (do not use this if you are varying the white noise!)

        rn_components: int
            number of red noise Fourier modes to include
            
        dm_components: int
            number of DM noise Fourier modes to include
        
        chrom_components: int
            number of chromatic noise Fourier modes to include
            
        dm_var: bool
            wheter to include a free spectrum gibbs dm_gp
        
        chrom_var: bool
            whether to include a free spectrum gibbs chrom_gp
            
        include_quadratic: bool
            whether or not to fit out a quadratic trend in chrom_gp (think DM2)
            
        log10rhomin: float
            lower bound for the log10 of the rho parameter.

        log10rhomax: float
            upper bound for the log10 of the rho parameter

        tnequad: string
            whether to use the temponest convension of efac and equad
        """

        self.psr = [psr]
        if Tspan:
            self.Tspan = Tspan
        else:
            self.Tspan = model_utils.get_tspan(self.psr)
        self.name = self.psr[0].name
        self.inc_ecorr = inc_ecorr
        self.ecorr_type = ecorr_type
        self.vary_wn = vary_wn
        self.tm_marg = tm_marg
        self.wn_names = ["efac", "equad", "ecorr"]
        self.rhomin = log10rhomin
        self.rhomax = log10rhomax
        self.rn_components = rn_components
        self.dm_components = dm_components
        self.chrom_components = chrom_components
        self.vary_rn = vary_rn
        self.vary_dm = vary_dm
        self.vary_chrom = vary_chrom
        self.include_quadratic = include_quadratic
        self.low = 10 ** (2 * self.rhomin)
        self.high = 10 ** (2 * self.rhomax)

        # Making the pta object
        if self.tm_marg:
            tm = gp_signals.MarginalizingTimingModel(use_svd=True)
            if self.vary_wn:
                warnings.warn(
                    "***FYI: the timing model is marginalized for. This will slow down the WN sampling!!***"
                )
        else:
            tm = gp_signals.TimingModel(use_svd=True)

        if self.ecorr_type == "basis":
            wn = blocks.white_noise_block(
                vary=self.vary_wn,
                inc_ecorr=self.inc_ecorr,
                gp_ecorr=True,
                select=select,
                tnequad=tnequad,
            )
        else:
            wn = blocks.white_noise_block(
                vary=self.vary_wn,
                inc_ecorr=self.inc_ecorr,
                gp_ecorr=False,
                select=select,
                tnequad=tnequad,
            )

        if self.vary_rn:
            rn = blocks.red_noise_block(
                psd="spectrum",
                prior="log-uniform",
                Tspan=self.Tspan,
                #logmin=self.rhomin,
                #logmax=self.rhomax,
                components=self.rn_components,
                gamma_val=None,
            )
        
        if self.vary_dm:
            dm = blocks.dm_noise_block(
                gp_kernel='diag',
                psd='spectrum',
                prior='log-uniform',
                Tspan=self.Tspan,
                components=self.dm_components,
                gamma_val=None,
                coefficients=False
            )
        
        if self.vary_chrom:
            chrom = blocks.chromatic_noise_block(
                gp_kernel='diag',
                psd='spectrum',
                prior='log-uniform',
                idx=4,
                include_quadratic=self.include_quadratic,
                Tspan=self.Tspan,
                name='chrom',
                components=self.chrom_components,
            )
                
        s = tm + wn
        
        if self.vary_rn:
            s += rn
        if self.vary_dm:
            s += dm
        if self.vary_chrom:
            s += chrom
        
        self.pta = signal_base.PTA(
            [s(p) for p in self.psr],
            lnlikelihood=signal_base.LogLikelihoodDenseCholesky,
        )
        #print(self.pta.signals.keys())
        if not self.vary_wn:
            self.pta.set_default_params(noise_dict)
            self.Nmat = self.pta.get_ndiag(params={})[0]
            self.TNr = self.pta.get_TNr(params={})[0]
            self.TNT = self.pta.get_TNT(params={})[0]
        else:
            self.Nmat = None

        if self.inc_ecorr and "basis" in self.ecorr_type and self.vary_wn:
            # grabbing priors on ECORR params
            for ct, par in enumerate(self.pta.params):
                if "ecorr" in str(par):
                    ind = ct
            ecorr_priors = str(self.pta.params[ind].params[0])
            ecorr_priors = ecorr_priors.split("(")[1].split(")")[0].split(", ")
            self.ecorrmin, self.ecorrmax = (
                10 ** (2 * float(ecorr_priors[0].split("=")[1])),
                10 ** (2 * float(ecorr_priors[1].split("=")[1])),
            )
            #print(self.ecorrmin, self.ecorrmax)

        # Getting residuals
        self._residuals = self.pta.get_residuals()[0]
        ## FIXME : maybe don't cache this -- could lead to memory issues.
        # Intial guess for the model params
        self._xs = np.array([p.sample()
                            for p in self.pta.params], dtype=object)
        # Initializign the b-coefficients.
        # The shape is 2*rn_comp+2*dm_comp+2*chrom_comp if tm_marg = True
        # if tm_marg = False, 
        # then the shape is more because there are some tm params in there?
        self._b = np.zeros(self.pta.get_basis(self._xs)[0].shape[1])
        # when including dm and chromatic models, the b's are 
        # the concantenation of the red noise, dm, and chromatic noise fourier coefficients
        #print("len b: ", len(self._b))
        #print(self.pta.get_basis(self._xs)[0].shape)
        self.Tmat = self.pta.get_basis(params={})[0]
        self.phiinv = None
        # print(self._xs.shape)
        # print(self.pta.params)
        # print("dm", self.get_dm_param_indices)
        # print("chrom", self.get_chrom_param_indices)
        # print("rn:", self.get_rn_param_indices)
        # find basis indices of GW process
        ### jeremy : changing the below from gwid to rn_id and adding dm_id and chrom_id
        self.rn_id = []
        self.dm_id = []
        self.chrom_id = []
        ct = 0
        psigs = [sig for sig in self.pta.signals.keys() if self.name in sig]
        for sig in psigs:
            Fmat = self.pta.signals[sig].get_basis()
            if "red_noise" in self.pta.signals[sig].name:
                self.rn_id.append(ct + np.arange(0, Fmat.shape[1]))
                ct+=Fmat.shape[1]
            if "dm_gp" in self.pta.signals[sig].name:
                self.dm_id.append(ct + np.arange(0, Fmat.shape[1]))
                ct+=Fmat.shape[1]
            if "chrom_gp" in self.pta.signals[sig].name:
                self.chrom_id.append(ct + np.arange(0, Fmat.shape[1]))
                ct+=Fmat.shape[1]
            ### jeremy : chaning the above to red_noise and adding dm and chrom as well
            # Avoid None-basis processes.
            # Also assume red + GW signals share basis.
            if Fmat is not None and "red" not in sig and 'dm_gp' not in sig and 'chrom_gp' not in sig:
                ct += Fmat.shape[1]
            #print(sig)
            #print(ct)
            #print("rn", self.rn_id)
            #print("dm", self.dm_id)
            #print("chrom", self.chrom_id)

    @cached_property
    def params(self):
        return self.pta.params

    @cached_property
    def param_names(self):
        return self.pta.param_names

    def map_params(self, xs):
        return self.pta.map_params(xs)

    @cached_property
    def get_rn_param_indices(self):
        ind = []
        for ct, par in enumerate(self.param_names):
            if "red_noise" in par:
                ind.append(ct)
        return np.array(ind)
    
    @cached_property
    def get_dm_param_indices(self):
        ind = []
        for ct, par in enumerate(self.param_names):
            if "dm_gp" in par:
                ind.append(ct)
        return np.array(ind)
    
    @cached_property
    def get_chrom_param_indices(self):
        ind = []
        for ct, par in enumerate(self.param_names):
            if "chrom_gp" in par:
                ind.append(ct)
        return np.array(ind)

    @cached_property
    def get_efacequad_indices(self):
        ind = []
        if "basis" in self.ecorr_type:
            for ct, par in enumerate(self.param_names):
                if "efac" in par or "equad" in par:
                    ind.append(ct)
        else:
            for ct, par in enumerate(self.param_names):
                if "ecorr" in par or "efac" in par or "equad" in par:
                    ind.append(ct)
        return np.array(ind)

    @cached_property
    def get_basis_ecorr_indices(self):
        ind = []
        for ct, par in enumerate(self.param_names):
            if "ecorr" in par:
                ind.append(ct)
        return np.array(ind)

    def update_red_params(self, xs):
        """
        Function to perform red_noise_log10_rho updates given 
        the red noise Fourier coefficients.
        """
        tau = self._b[tuple(self.rn_id)] ** 2
        tau = (tau[0::2] + tau[1::2]) / 2

        Norm = 1 / (np.exp(-tau / self.high) - np.exp(-tau / self.low))
        x = np.random.default_rng().uniform(0, 1, size=tau.shape)
        rhonew = -tau / np.log(x / Norm + np.exp(-tau / self.low))
        xs[-1] = 0.5 * np.log10(rhonew)
        return xs

    def update_dm_params(self, xs):
        """
        Function to perform dm_gp_log10_rho updates given 
        the dm gp Fourier coefficients.
        """
        tau = self._b[tuple(self.dm_id)] ** 2
        tau = (tau[0::2] + tau[1::2]) / 2

        Norm = 1 / (np.exp(-tau / self.high) - np.exp(-tau / self.low))
        x = np.random.default_rng().uniform(0, 1, size=tau.shape)
        rhonew = -tau / np.log(x / Norm + np.exp(-tau / self.low))
        xs[-2] = 0.5 * np.log10(rhonew)
        return xs
    
    def update_chrom_params(self, xs):
        """
        Function to perform chrom_gp_log10_rho updates given 
        the chromatic gp Fourier coefficients.
        """
        tau = self._b[tuple(self.chrom_id)] ** 2
        tau = (tau[0::2] + tau[1::2]) / 2

        Norm = 1 / (np.exp(-tau / self.high) - np.exp(-tau / self.low))
        x = np.random.default_rng().uniform(0, 1, size=tau.shape)
        rhonew = -tau / np.log(x / Norm + np.exp(-tau / self.low))
        xs[-3] = 0.5 * np.log10(rhonew)
        return xs

    def update_b(self, xs):
        """
        Function to perform updates on Fourier coefficients given other model parameters.
        """
        params = self.pta.map_params(np.hstack(xs))
        self._phiinv = self.pta.get_phiinv(params, logdet=False)[0]

        try:
            TNT = self.TNT.copy()
        except BaseException:
            T = self.Tmat
            TNT = self.Nmat.solve(T, left_array=T)
        try:
            TNr = self.TNr.copy()
        except BaseException:
            T = self.Tmat
            TNr = self.Nmat.solve(self._residuals, left_array=T)

        np.fill_diagonal(TNT, TNT.diagonal() + self._phiinv)
        try:
            chol = cho_factor(
                TNT,
                lower=True,
                overwrite_a=False,
                check_finite=False)
            mean = cho_solve(
                chol,
                b=TNr,
                overwrite_b=False,
                check_finite=False)
            self._b = mean + st_solve(
                chol[0],
                np.random.normal(loc=0, scale=1, size=TNT.shape[0]),
                lower=True,
                unit_diagonal=False,
                overwrite_b=False,
                check_finite=False,
                trans=1,
            )
        except np.linalg.LinAlgError:
            print("oh sh******t; a spiiiiiddddeeeerrrrrr")
            if self.bchain.any():
                self._b = self.bchain[
                    np.random.default_rng().integers(0, len(self.bchain))
                ]
            else:
                bchain = np.memmap(
                    self._savepath + "/chain_1",
                    dtype="float32",
                    mode="r",
                    shape=(self.niter, self.len_x + self.len_b),
                )[:, -len(self._b):]
                self._b = bchain[np.random.default_rng().integers(
                    0, len(bchain))]

    def update_white_params(self, xs, iters=10):
        """
        Function to perform WN updates given other model parameters.
        If kernel ecorr is chosen, WN includes ecorr as well.
        """
        # get white noise parameter indices
        wind = self.get_efacequad_indices
        xnew = xs
        x0 = xnew[wind].copy()
        lnlike0, lnprior0 = self.get_lnlikelihood_white(
            x0), self.get_wn_lnprior(x0)
        lnprob0 = lnlike0 + lnprior0

        for ii in range(
                self.start_wn_iter + 1,
                self.start_wn_iter + iters + 1):
            x0, lnlike0, lnprob0 = self.sampler_wn.PTMCMCOneStep(
                x0, lnlike0, lnprob0, ii
            )
        xnew[wind] = x0
        self.start_wn_iter = ii

        # Do some caching of "later needed" parameters for improved performance
        self.Nmat = self.pta.get_ndiag(self.map_params(xnew))[0]
        Tmat = self.Tmat
        if "basis" not in self.ecorr_type:
            self.TNT = self.Nmat.solve(Tmat, left_array=Tmat)
        else:
            TN = Tmat / self.Nmat[:, None]
            self.TNT = Tmat.T @ TN
            residuals = self._residuals
            self.rNr = np.sum(residuals**2 / self.Nmat)
            self.logdet_N = np.sum(np.log(self.Nmat))
            self.d = TN.T @ residuals

        return xnew

    def update_basis_ecorr_params(self, xs, iters=10):
        """
        Function to perform basis ecorr updates.
        """
        # get white noise parameter indices
        eind = self.get_basis_ecorr_indices
        xnew = xs
        x0 = xnew[eind].copy()
        lnlike0, lnprior0 = self.get_basis_ecorr_lnlikelihood(
            x0
        ), self.get_basis_ecorr_lnprior(x0)
        lnprob0 = lnlike0 + lnprior0

        for ii in range(
                self.start_ec_iter + 1,
                self.start_ec_iter + iters + 1):
            x0, lnlike0, lnprob0 = self.sampler_ec.PTMCMCOneStep(
                x0, lnlike0, lnprob0, ii
            )
        xnew[eind] = x0
        self.start_ec_iter = ii

        return xnew

    def get_lnlikelihood_white(self, xs):
        """
        Function to calculate WN log-liklihood.
        """
        x0 = self._xs.copy()
        x0[self.get_efacequad_indices] = xs

        params = self.map_params(x0)
        Nmat = self.pta.get_ndiag(params)[0]
        # whitened residuals
        yred = self._residuals - self.Tmat @ self._b
        try:
            if "basis" not in self.ecorr_type:
                rNr, logdet_N = Nmat.solve(yred, left_array=yred, logdet=True)
            else:
                rNr = np.sum(yred**2 / Nmat)
                logdet_N = np.sum(np.log(Nmat))
        except BaseException:
            return -np.inf
        # first component of likelihood function
        loglike = -0.5 * (logdet_N + rNr)

        return loglike

    def get_basis_ecorr_lnlikelihood(self, xs):
        """
        Function to calculate basis ecorr log-liklihood.
        """
        x0 = np.hstack(self._xs.copy())
        x0[self.get_basis_ecorr_indices] = xs

        params = self.map_params(x0)
        # start likelihood calculations
        loglike = 0
        # get auxiliaries
        phiinv, logdet_phi = self.pta.get_phiinv(params, logdet=True)[0]
        # first component of likelihood function
        loglike += -0.5 * (self.logdet_N + self.rNr)
        # Red noise piece
        Sigma = self.TNT + np.diag(phiinv)
        try:
            cf = sl.cho_factor(Sigma)
            expval = sl.cho_solve(cf, self.d)
        except np.linalg.LinAlgError:
            return -np.inf

        logdet_sigma = np.sum(2 * np.log(np.diag(cf[0])))
        loglike += 0.5 * (self.d @ expval - logdet_sigma - logdet_phi)

        return loglike

    def get_wn_lnprior(self, xs):
        """
        Function to calculate WN log-prior.
        """
        x0 = self._xs.copy()
        x0[self.get_efacequad_indices] = xs

        return np.sum([p.get_logpdf(value=x0[ct])
                      for ct, p in enumerate(self.params)])

    def get_basis_ecorr_lnprior(self, xs):
        """
        Function to calculate basis ecorr log-prior.
        """
        x0 = self._xs.copy()
        x0[self.get_basis_ecorr_indices] = xs

        return np.sum([p.get_logpdf(value=x0[ct])
                      for ct, p in enumerate(self.params)])

    def sample(
        self,
        niter=int(1e4),
        wniters=100,
        eciters=15,
        savepath=None,
        SCAMweight=30,
        AMweight=15,
        DEweight=50,
        covUpdate=1000,
        burn=10000,
        **kwargs
    ):
        """
        Gibbs Sampling

        Parameters
        -----------
        niter: integer
            total number of Gibbs sampling iterations

        wniters:
            number of white noise MCMC sampling iterations within each Gibbs step

        eciters:
            number of basis ecorr MCMC sampling iterations within each Gibbs step

        savepath: str
            the path to save the chains

        covUpdate: integer
            Number of iterations between AM covariance updates

        SCAMweight: integer
            Weight of SCAM jumps in overall jump cycle

        AMweight: integer
            Weight of AM jumps in overall jump cycle

        DEweight: integer
            Weight of DE jumps in overall jump cycle

        kwargs: dict
            PTMCMC initialization settings not mentioned above
        """
        self.start_wn_iter = 0
        self.start_ec_iter = 0

        os.makedirs(savepath, exist_ok=True)

        if self.vary_wn:
            # large number to avoid saving the white noise choice in a txt file
            isave = int(4e9)
            thin = 1
            Niter = int(niter * wniters + 1)

            x0 = self._xs[self.get_efacequad_indices]
            ndim = len(x0)
            cov = np.diag(
                np.ones(ndim) * 0.01**2
            )  # helps to tune MCMC proposal distribution
            self.sampler_wn = ptmcmc(
                ndim,
                self.get_lnlikelihood_white,
                self.get_wn_lnprior,
                cov,
                outDir=savepath,
                resume=False,
            )
            self.sampler_wn.initialize(
                Niter=Niter,
                isave=isave,
                thin=thin,
                SCAMweight=SCAMweight,
                AMweight=AMweight,
                DEweight=DEweight,
                covUpdate=covUpdate,
                burn=burn,
                **kwargs
            )

            if "basis" in self.ecorr_type and self.vary_wn and self.inc_ecorr:
                x0 = self._xs[self.get_basis_ecorr_indices]
                ndim = len(x0)
                cov = np.diag(np.ones(ndim) * 0.01**2)
                self.sampler_ec = ptmcmc(
                    ndim,
                    self.get_basis_ecorr_lnlikelihood,
                    self.get_basis_ecorr_lnprior,
                    cov,
                    outDir=savepath,
                    resume=False,
                )
                self.sampler_ec.initialize(
                    Niter=Niter,
                    isave=isave,
                    thin=thin,
                    SCAMweight=SCAMweight,
                    AMweight=AMweight,
                    DEweight=DEweight,
                    covUpdate=covUpdate,
                    burn=burn,
                    **kwargs
                )

        np.savetxt(savepath + "/pars.txt",
                   list(map(str, self.pta.param_names)), fmt="%s")
        np.savetxt(
            savepath + "/priors.txt",
            list(map(lambda x: str(x.__repr__()), self.pta.params)),
            fmt="%s",
        )
        if self.vary_rn:
            rn_freqs = np.arange(
                1 / self.Tspan,
                (self.rn_components + 0.001) / self.Tspan,
                1 / self.Tspan)
            np.save(savepath + "/rn_freqs.npy", rn_freqs)
        
        if self.vary_dm:
            dm_freqs = np.arange(
                1 / self.Tspan,
                (self.dm_components + 0.001) / self.Tspan,
                1 / self.Tspan)
            np.save(savepath + "/dm_freqs.npy", dm_freqs)
        if self.vary_chrom:
            chrom_freqs = np.arange(
                1 / self.Tspan,
                (self.chrom_components + 0.001) / self.Tspan,
                1 / self.Tspan)
            np.save(savepath + "/chrom_freqs.npy", chrom_freqs)
        [os.remove(dpa) for dpa in glob.glob(savepath + "/*jump.txt")]

        xnew = self._xs.copy()

        len_b = len(self._b)
        len_x = len(np.hstack(self._xs))
        self._savepath = savepath

        fp = np.lib.format.open_memmap(
            savepath + "/chain_1.npy",
            mode="w+",
            dtype="float32",
            shape=(niter, len_x + len_b),
            fortran_order=False,
        )

        pbar = tqdm(range(niter), colour="GREEN")
        pbar.set_description("Sampling %s" % self.name)
#        num_gibbs = np.sum([int(self.vary_rn), int(self.vary_dm), int(self.vary_chrom)])
        for ii in pbar:
            if self.vary_wn:
                xnew = self.update_white_params(xnew, iters=wniters)

            if self.inc_ecorr and "basis" in self.ecorr_type:
                xnew = self.update_basis_ecorr_params(xnew, iters=eciters)

#            turn = ii % num_gibbs
            #if self.vary_rn and turn == 0:
            self.update_b(xs=xnew)
            xnew = self.update_red_params(xs=xnew)
            #if self.vary_dm and turn == 1:
                #self.update_b(xs=xnew)
            xnew = self.update_dm_params(xs=xnew)
            #if self.vary_chrom and turn == 2:
                #self.update_b(xs=xnew)
            xnew = self.update_chrom_params(xs=xnew)
            
            fp[ii, -len_b:] = self._b
            fp[ii, 0:len_x] = np.hstack(xnew)

