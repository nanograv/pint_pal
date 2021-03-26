# Generic imports
import os, sys, glob, tempfile, pickle
import numpy as np
import scipy.linalg as sl, scipy.optimize as so
import matplotlib.pyplot as plt
import numdifftools as nd
import corner

# Non-traditional packages
import pint.toa as toa
import pint.models as model
from enterprise.pulsar import PintPulsar

# The actual outlier code
import interval as itvl
from nutstrajectory import nuts6

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
    !mkdir -p {chaindir}

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
