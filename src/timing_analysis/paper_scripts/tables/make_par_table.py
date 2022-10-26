#! /usr/bin/env python
#%%

# Make tex table of pulsar and TOA params for paper
#
# Updated (quite drastically) from 12.5yr script to use PINT+Python3,
# DeluxeTable environment, and handle another frequency column and many
# more PSRs, on Sept 3, 2022, by Scott Ransom

import os, os.path, sys, glob
import numpy as np
import pint.toa as pt
import pint.models as pm
import pint.logging

pint.logging.setup(level="INFO")

release_path = "20220314.Release.nb.78afc797"
pars = glob.glob(f"{release_path}/*.par")
tims = glob.glob(f"{release_path}/*.tim")
tmpPSRs = [os.path.split(x)[1].split("_")[0] for x in pars]
# Now remove the pulsar names that end in letters (i.e. observatories)
PSRs = sorted([x for x in tmpPSRs if x[-1].isnumeric()])

#%%
# Now load the tim files and the parfiles
data = {}
for PSR in PSRs:
    # The following remove the tim/par files that have "gbt" or "ao" after the pulsar name
    parfile = [x for x in pars if f"{PSR}_" in x][0]
    timfile = [x for x in tims if f"{PSR}_" in x][0]
    data[PSR] = pm.get_model_and_toas(parfile, timfile, usepickle=True)

#%%
if False: # If true, this will determine the receivers and backends
    receivers = set([])
    backends = set([])
    for PSR in PSRs:
        toas = data[PSR][1]
        vals, inds = toas.get_flag_value("fe")
        receivers |= set(np.unique(vals))
        vals, inds = toas.get_flag_value("be")
        backends |= set(np.unique(vals))
else:
    receivers = {'327', '430', 'Rcvr_800', 'L-wide', 'Rcvr1_2', '1.5GHz', 'S-wide', '3GHz'}
    backends = {'ASP', 'GASP', 'GUPPI', 'PUPPI', 'YUPPI'}

#%%
import astropy.units as u

bw0 = 100.0 # nomininal TOA BW in MHz
t0 = 1800.0 # nomininal TOA integration in sec

def scale_errors(toas):
    "Return properly scaled TOA errors as a numpy array"
    errs = toas.get_errors().to(u.us).value # in microsec
    bws = np.asarray(toas.get_flag_value('bw', as_type=float)[0])
    ts = np.asarray(toas.get_flag_value('tobs', as_type=float)[0])
    return errs * np.sqrt(bws * ts / (bw0 * t0))

hdr = r'''
\startlongtable
\begin{deluxetable*}{crrrrr@{}lr@{}lr@{}lr@{}lr@{}lr@{}lr}
\tabletypesize{\footnotesize}
%\tablewidth{dimen}
%\tablenum{text}
%\tablecolumns{12}
\tablecaption{Basic Pulsar Parameters and TOA Statistics\label{tab:psrtoastats}}
\tablehead{
    \colhead{Source} &
    \colhead{$P$} &
    \colhead{$dP/dt$} &
    \colhead{DM} &
    \colhead{$P_b$} &
    \multicolumn{12}{c}{Median scaled TOA uncertainty\tablenotemark{a} ($\mu$s) / Number of epochs} &
    \colhead{Span} \\
    \cline{6-17} &
    \colhead{(ms)} &
    \colhead{($10^{-20}$)} &
    \colhead{(pc\,cm$^{-3}$)} &
    \colhead{(d)} &
    \multicolumn{2}{c}{327\,MHz} &
    \multicolumn{2}{c}{430\,MHz} &
    \multicolumn{2}{c}{820\,MHz} &
    \multicolumn{2}{c}{1.4\,GHz} &
    \multicolumn{2}{c}{2.1\,GHz} &
    \multicolumn{2}{c}{3.0\,GHz} &
    \colhead{(yr)}}
\startdata
'''

# See Table 1 of 9yr dataset paper for nominal BWs
# The scaling factors are sqrt(nomBM/100MHz)
# Freq columns are 327, 430, 820, 1.4, 2.1, 3.0
scalings = f'''\multicolumn{{5}}{{r}}{{Nominal scaling factors\\tablenotemark{{b}} for ASP/GASP:}}
  & \multicolumn{{2}}{{c}}{{{np.sqrt(34/bw0):.2f}}}
  & \multicolumn{{2}}{{c}}{{{np.sqrt(20/bw0):.2f}}}
  & \multicolumn{{2}}{{c}}{{{np.sqrt(64/bw0):.2f}}}
  & \multicolumn{{2}}{{c}}{{{np.sqrt(64/bw0):.2f}}}
  & \multicolumn{{2}}{{c}}{{{np.sqrt(64/bw0):.2f}}}
  & \multicolumn{{2}}{{c}}{{$\cdots$}}
  & \\\\
\multicolumn{{5}}{{r}}{{GUPPI/PUPPI:}}
  & \multicolumn{{2}}{{c}}{{{np.sqrt(50/bw0):.2f}}}
  & \multicolumn{{2}}{{c}}{{{np.sqrt(24/bw0):.2f}}}
  & \multicolumn{{2}}{{c}}{{{np.sqrt(186/bw0):.2f}}}
  & \multicolumn{{2}}{{c}}{{{np.sqrt(620/bw0):.2f}}}
  & \multicolumn{{2}}{{c}}{{{np.sqrt(460/bw0):.2f}}}
  & \multicolumn{{2}}{{c}}{{$\cdots$}}
  & \\\\
\multicolumn{{5}}{{r}}{{YUPPI:}}
  & \multicolumn{{2}}{{c}}{{$\cdots$}}
  & \multicolumn{{2}}{{c}}{{$\cdots$}}
  & \multicolumn{{2}}{{c}}{{$\cdots$}}
  & \multicolumn{{2}}{{c}}{{{np.sqrt(600/bw0):.2f}}}
  & \multicolumn{{2}}{{c}}{{$\cdots$}}
  & \multicolumn{{2}}{{c}}{{{np.sqrt(1500/bw0):.2f}}}
  & \\\\
'''

foot = r'''\enddata
\tablenotetext{a}{Original TOA uncertainties were scaled by their
bandwidth-time product $\left( \frac{\Delta \nu}{100\,\mathrm{MHz}}
\frac{\tau}{1800\,\mathrm{s}} \right)^{1/2}$ to remove variation due to
different instrument bandwidths and integration times.}
\tablenotetext{b}{TOA uncertainties can be rescaled to the nominal full
instrumental bandwidth by dividing by these scaling factors.}
\end{deluxetable*}
'''
#%%
# Get the basic pulsar information
lines = {}
for PSR in PSRs:
    m = data[PSR][0] # timing model
    f0 = m.F0.value
    f1 = m.F1.value
    p = f'{1e3/f0:.2f}'
    pdot = f'{-1e20*f1/(f0*f0):.2f}'
    dm = f'{m.DM.value:.1f}'
    if m.is_binary:
        # All binaries have PB, but if using FB0, then PB is unset
        x =  m.PB.value or m.FB0.value**-1 / (3600*24)
        pb = f'{x:.1f}'
        # Not used, but keeping this in
        tmax = f'{min(x*1440.0*0.02, 30.0):.0f}'
    else:
        pb = '$\cdots$'
        tmax = '30'
    tex_psr = PSR.replace('-','$-$').replace('+','$+$')
    if tex_psr.startswith('B'): tex_psr += '\phantom{....}'
    lines[PSR] = ' & '.join((tex_psr,p,pdot,dm,pb))

#%%
# This is better than the old method which rounded MJDs
# as that can combine multiple obs in the same day
def count_epochs(toas, rcvr_inds):
    cs = toas.get_clusters()
    return sum(bool((rcvr_inds * (cs == ii)).sum()) for ii in range(cs.max()))

# compute the TOA information
for PSR in PSRs:
    toas = data[PSR][1] # TOAs
    rcvr = np.asarray(toas.get_flag_value('fe')[0])
    errs = toas.get_errors()
    mjds = toas.get_mjds()
    errs_scl = scale_errors(toas)
    idx = {'327': rcvr == '327',
           '430': rcvr == '430',
           '820': rcvr == 'Rcvr_800',
           '1400': (rcvr == 'Rcvr1_2') | (rcvr == 'L-wide') | (rcvr == '1.5GHz'),
           '2100': rcvr == 'S-wide',
           '3000': rcvr == '3GHz'}
    sig_t = {}
    for k, v in idx.items():
        n_ep = count_epochs(toas, v)
        if v.sum(): # there are TOAs for that receiver
            med_err = np.median(errs_scl[v])
            # sig_t[k] = f'{med_err:.3f} / {n_ep}'
            sig_t[k] = f'{med_err:.3f} &/{n_ep}'
        else:
            #sig_t[k] = '$\cdots$'
            sig_t[k] = '\multicolumn{2}{c}{$\cdots$}'
    for k in ('327','430','820','1400','2100','3000'):
        lines[PSR] += f' & {sig_t[k]}'
    lines[PSR] += f' & {(mjds.max() - mjds.min()).value/365.24:.1f} \\\\'

#%%
with open("table_params+toastats.tex", "w") as f:
    f.write(hdr)
    # Sort the lines based on the RA numbers, not "J" or "B"
    for PSR in sorted(PSRs, key=lambda x: x[1:]):
        f.write(lines[PSR]+"\n")
    f.write(scalings)
    f.write(foot)

# %%
