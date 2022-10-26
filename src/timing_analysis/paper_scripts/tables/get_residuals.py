#! /usr/bin/env python
#%%
import os, os.path, glob, pickle
import numpy as np
import pint.toa as pt
import pint.models as pm
import pint.fitter as pf
import pint.logging
import utils # this is code taken from timing_analysis.utils

pint.logging.setup(level="INFO")

release_path = "20220314.Release.nb.78afc797"
pars = glob.glob(f"{release_path}/*.par")
tims = glob.glob(f"{release_path}/*.tim")
tmpPSRs = [os.path.split(x)[1].split("_")[0] for x in pars]
# Now remove the pulsar names that end in letters (i.e. observatories)
PSRs = sorted([x for x in tmpPSRs if x[-1].isnumeric()])

#%%
rmss = {}
for PSR in PSRs:
    # The following remove the tim/par files that have "gbt" or "ao" after the pulsar name
    parfile = [x for x in pars if f"{PSR}_" in x][0]
    timfile = [x for x in tims if f"{PSR}_" in x][0]
    pint.logging.log.warning(f"Working on {PSR}")
    # Now load the tim files and the parfiles
    m, t = pm.get_model_and_toas(parfile, timfile, usepickle=True)
    f = pf.DownhillGLSFitter(t, m)
    f.fit_toas()
    vals = {"raw": utils.resid_stats(f, epoch_avg=True, whitened=False)}
    vals["white"] = utils.resid_stats(f, epoch_avg=True, whitened=True)
    vals["ntoas"] = t.ntoas
    rmss[PSR] = vals

with open("weighted_rmss_20220314.pickle", "wb") as f:
    pickle.dump(rmss, f)

    
