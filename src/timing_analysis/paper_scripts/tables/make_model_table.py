#! /usr/bin/env python
#%%

# Make tex table of pulsar params for paper
#
# Updated (quite drastically) from 12.5yr script to use PINT+Python3,
# DeluxeTable environment, and handle many more PSRs, 
# on Sept 10, 2022, by Scott Ransom

import glob, os, sys, pickle
import numpy as np
import pint.models as pm
import pint.logging

pint.logging.setup(level="INFO")

hdr = r'''
\startlongtable
\begin{deluxetable*}{cr|rrrrrr|cc|ccc|c}
\tabletypesize{\footnotesize}
%\tablewidth{dimen}
%\tablenum{text}
%\tablecolumns{14}
\tablecaption{Summary of Timing Model Fits\label{tab:models}}
\tablehead{
    \colhead{\phantom{XX}Source\phantom{XX}} &
    \colhead{\phantom{X}Number\phantom{X}} &
    \multicolumn{6}{c}{\phantom{XX}Number of Fit Parameters\tablenotemark{a}\phantom{XX}} &
    \multicolumn{2}{c}{\phantom{XX}rms\tablenotemark{b} ($\mu$s)\phantom{XX}} &
    \multicolumn{3}{c}{\phantom{XX}Red Noise\tablenotemark{c}\phantom{XX}} &
    \colhead{\phantom{XX}Figure\phantom{XX}} \\
    \cline{3-8} \cline{9-10} \cline{11-13}
    \colhead{\phantom{XXSourceXX}} &
    \colhead{of TOAs} &
    \colhead{S\phantom{X}} &
    \colhead{A\phantom{X}} &
    \colhead{B\phantom{X}} &
    \colhead{DM} &
    \colhead{FD} &
    \colhead{J\phantom{X}} &
    \colhead{Full} &
    \colhead{White} &
    \colhead{$A_{\mathrm{red}}$} &
    \colhead{$\gamma_{\mathrm{red}}$} &
    \colhead{log$_{10}B$} &
    \colhead{Number}}
\startdata
'''

foot = r'''\enddata
\tablenotetext{a}{Fit parameters: S=spin; A=astrometry; B=binary;
DM=dispersion measure; FD=frequency dependence; J=jump}
\tablenotetext{b}{Weighted root-mean-square of epoch-averaged post-fit
timing residuals, calculated using the procedure described in Appendix D
of \nineyr. For sources with red noise, the ``Full'' rms value includes
the red noise contribution, while the ``White'' rms does not.}
\tablenotetext{c}{Red noise parameters: $A_{\mathrm{red}}$ = amplitude of
red noise spectrum at $f$=1~yr$^{-1}$ measured in $\mu$s yr$^{1/2}$;
$\gamma_{\mathrm{red}}$ = spectral index; $B$ = Bayes factor (``$>$2''
indicates a Bayes factor larger than our threshold log$_{10}$B~$>$~2, but
which could not be estimated using the Savage-Dickey ratio).  See
Eqn.~\ref{eqn:rn_spec} and Appendix~C of \nineyr\ for details.}
\end{deluxetable*}
'''

# Generated using read_noise_chains.py
with open("noise_results_20220314.pickle", "rb") as f:
    nr = pickle.load(f)

# Generated using get_residuals.py
with open("weighted_rmss_20220314.pickle", "rb") as f:
    rmss = pickle.load(f)

release_path = "20220314.Release.nb.78afc797"
pars = glob.glob(f"{release_path}/*.par")
tims = glob.glob(f"{release_path}/*.tim")
tmpPSRs = [os.path.split(x)[1].split("_")[0] for x in pars]
# Now remove the pulsar names that end in letters (i.e. observatories)
tmpPSRs = [x for x in tmpPSRs if x[-1].isnumeric()]
# And now sort by the RA string
PSRs = sorted(tmpPSRs, key=lambda x: x[1:])
assert(len(PSRs)==len(rmss.keys()))

#%%

# Start making the data lines with the PSR and number of TOAs
lines = {}
for PSR in PSRs:
    tex_psr = PSR.replace('-','$-$').replace('+','$+$')
    if tex_psr.startswith('B'): tex_psr += '\phantom{....}'
    lines[PSR] = f'{tex_psr} & {rmss[PSR]["ntoas"]} & '

#%%

astrometric = {"RAJ", "DECJ", "PMRA", "PMDEC", "PX",
               "ELAT", "ELONG", "PMELAT", "PMELONG",
               "LAMBDA", "BETA", "PMLAMBDA", "PMBETA"}
spin = {"F", "F0", "F1", "F2", "F3", "F4"}

# Read the timing models in and separate params
for PSR in PSRs:
    parfile = [x for x in pars if f"{PSR}_" in x][0]
    m = pm.get_model(parfile)
    fitted = set(m.free_params)
    dmxs = {x for x in fitted if x.startswith("DMX_")}
    fitted -= dmxs
    fds = {x for x in fitted if x.startswith("FD")}
    fitted -= fds
    jumps = {x for x in fitted if x.startswith("JUMP")}
    fitted -= jumps
    astrom = fitted.intersection(astrometric)
    fitted -= astrom
    spin = fitted.intersection(spin) # remember that PHASE is also fit!
    fitted -= spin
    # Assume anything left is binary
    binary = fitted
    lines[PSR] += f"{len(spin)+1} & {len(astrom)} & {len(binary)} & {len(dmxs)} & {len(fds)} & {len(jumps)} & "

#%%

# Now add the RMS, rednoise, and figure info
for PSR in PSRs:
    lines[PSR] += f"{rmss[PSR]['raw']['All']['wrms'].value:.3f} & "
    bf = nr[PSR][1] # This is the rednoise Bayes Factor
    if np.isnan(bf): # We have strong red noise
        lines[PSR] += f"{rmss[PSR]['white']['All']['wrms'].value:.3f} & "
        gamma = -nr[PSR][0][f"{PSR}_red_noise_gamma"]
        rnamp = 10**nr[PSR][0][f"{PSR}_red_noise_log10_A"] / 1e-12
        lines[PSR] += f"{rnamp:.3f} & {gamma:.1f} & $>$2 & \\ref{{fig:summary-{PSR}}} \\\\"
    else:
        lines[PSR] += f"$\cdots$ & $\cdots$ & $\cdots$ & {np.log10(bf):.2f} & \\ref{{fig:summary-{PSR}}} \\\\"

#%%

with open("table_model_summaries.tex", "w") as f:
    f.write(hdr)
    for PSR in PSRs:
        f.write(lines[PSR]+"\n")
    f.write(foot)

# %%
