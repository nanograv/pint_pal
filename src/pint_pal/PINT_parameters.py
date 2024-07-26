import pint.models.parameter as p
import astropy.units as u
from pint import ls

"""
This file will be a list of pre-defined PINT parameters and components to be used for F-tests.
Current parameters to be listed:

PX, PMLAMBDA, PMBETA, PMRA, PMDEC, H3, H4, K96, M2, SINI, PBDOT, XDOT, EPS1DOT, EPS2DOT, OMDOT, EDOT, FBX, FDX, F1, F2, F3

TO DO: Check units, check parameters/components are appropriately set for different binary models.
-> May want to manually add the binary model prefix to each parameter appropriately e.g. component = 'BINARY', add 'DD' before F-Test.

Those are the default values in the stand alone binary models
"PB": np.longdouble(10.0) u.day,
"PBDOT": 0.0 u.day / u.day,
"ECC": 0.0 u.Unit(""),
"EDOT": 0.0 / u.second,
"A1": 10.0 ls,
"A1DOT": 0.0 ls / u.second,
"T0": np.longdouble(54000.0) u.day,
"OM": 0.0 u.deg,
"OMDOT": 0.0 u.deg / u.year,
"XPBDOT": 0.0 u.day / u.day,
"M2": 0.0 u.M_sun,
"SINI": 0 u.Unit(""),
"GAMMA": 0 u.second,
"FB0": 1.1574e-6 * u.Unit("") / u.second,
These are DD specific
"A0": 0 u.second,
"B0": 0 u.second,
"DR": 0 u.Unit(""),
"DTH": 0 u.Unit("")
"""

# Parallax
PX = p.floatParameter(parameter_type="float",
    name="PX",
    value=0.0,
    units=u.mas,
    frozen = False,
    convert_tcb2tdb=False)
PX_Component = 'AstrometryEcliptic'

# Proper Motion
PMLAMBDA = p.floatParameter(parameter_type="float",
    name="PMLAMBDA",
    value=0.0,
    units=u.mas/u.yr,
    frozen = False,
    convert_tcb2tdb=False)
PMLAMBDA_Component = 'AstrometryEcliptic'

PMBETA = p.floatParameter(parameter_type="float",
    name="PMBETA",
    value=0.0,
    units=u.mas/u.yr,
    frozen = False,
    convert_tcb2tdb=False)
PMBETA_Component = 'AstrometryEcliptic'

PMRA = p.floatParameter(parameter_type="float",
    name="PMRA",
    value=0.0,
    units=u.mas/u.yr,
    frozen = False,
    convert_tcb2tdb=False)
PMRA_Component = 'AstrometryEcliptic'

PMDEC = p.floatParameter(parameter_type="float",
    name="PMDEC",
    value=0.0,
    units=u.mas/u.yr,
    frozen = False,
    convert_tcb2tdb=False)
PMDEC_Component = 'AstrometryEcliptic'

# H3 and H4
H3 = p.floatParameter(parameter_type="float",
    name="H3",
    value=0.0,
    units=u.s,
    frozen=False,
    convert_tcb2tdb=False)
H3_Component = 'Binary'

H4 = p.floatParameter(parameter_type="float",
    name="H4",
    value=0.0,
    units=u.s,
    frozen=False,
    convert_tcb2tdb=False)
H4_Component = 'Binary'

# K96, turns on flag for Kopeikin binary model proper motion correction
K96 = p.prefixParameter(parameter_type="bool",
    name="K96",
    value=True,
    convert_tcb2tdb=False)
K96_Component = 'Binary'

# M2 and SINI -> check units
M2 = p.floatParameter(parameter_type="float",
    name="M2",
    value=0.25,
    units=u.solMass,
    frozen=False,
    convert_tcb2tdb=False)
M2_Component = 'Binary'

SINI = p.floatParameter(parameter_type="float",
    name="SINI",
    value=0.8,
    units="",
    frozen=False,
    convert_tcb2tdb=False)
SINI_Component = 'Binary'

# PBDOT
PBDOT = p.floatParameter(parameter_type="float",
    name="PBDOT",
    value=0.0,
    units="",
    frozen=False,
    convert_tcb2tdb=False)
PBDOT_Component = 'Binary'

# XDOT
XDOT = p.floatParameter(parameter_type="float",
    name="XDOT",
    value=0.0,
    units="",
    frozen = False,
    convert_tcb2tdb=False)
XDOT_Component = 'Binary'

# A1DOT
A1DOT = p.floatParameter(parameter_type="float",
    name="A1DOT",
    value=0.0,
    units= ls / u.second,
    frozen = False,
    convert_tcb2tdb=False)
A1DOT_Component = 'Binary'

#EPS1DOT and EPS2DOT
EPS1DOT = p.floatParameter(parameter_type="float",
    name="EPS1DOT",
    value=0.0,
    units=1e-12/u.s,
    frozen = False,
    convert_tcb2tdb=False)
EPS1DOT_Component = 'Binary'

EPS2DOT = p.floatParameter(parameter_type="float",
    name="EPS2DOT",
    value=0.0,
    units=1e-12/u.s,
    frozen = False,
    convert_tcb2tdb=False)
EPS2DOT_Component = 'Binary'

# OMDOT
OMDOT = p.floatParameter(parameter_type="float",
    name="OMDOT",
    value=0.0,
    units=(u.deg/u.year),
    frozen = False,
    convert_tcb2tdb=False)
OMDOT_Component = 'Binary'

# EDOT
EDOT = p.floatParameter(parameter_type="float",
    name="EDOT",
    value=0.0,
    units=(1/u.s),
    frozen = False,
    convert_tcb2tdb=False)
EDOT_Component = 'Binary'

# FBX -> Do we need more? Is there a better way to do this? Check these...
FB0 = p.prefixParameter(parameter_type="float",
    name="FB0",
    value=0.0,
    units=1/u.s,
    frozen = False,
    convert_tcb2tdb=False)
FB0_Component = 'Binary'

FB1 = p.prefixParameter(parameter_type="float",
    name="FB1",
    value=0.0,
    units=1/u.s/u.s,
    frozen = False,
    convert_tcb2tdb=False)
FB1_Component = 'Binary'

FB2 = p.prefixParameter(parameter_type="float",
    name="FB2",
    value=0.0,
    units=1/u.s/u.s/u.s,
    frozen = False,
    convert_tcb2tdb=False)
FB2_Component = 'Binary'

FB3 = p.prefixParameter(parameter_type="float",
    name="FB3",
    value=0.0,
    units=1/u.s/u.s/u.s/u.s,
    frozen = False,
    convert_tcb2tdb=False)
FB3_Component = 'Binary'

FB4 = p.prefixParameter(parameter_type="float",
    name="FB4",
    value=0.0,
    units=1/u.s/u.s/u.s/u.s/u.s,
    frozen = False,
    convert_tcb2tdb=False)
FB4_Component = 'Binary'

FB5 = p.prefixParameter(parameter_type="float",
    name="FB5",
    value=0.0,
    units=1/u.s/u.s/u.s/u.s/u.s/u.s,
    frozen = False,
    convert_tcb2tdb=False)
FB5_Component = 'Binary'

# FDX -> Do we need more? Is there a better way to do this?
FD1 = p.prefixParameter(parameter_type="float",
    name="FD1",
    value=0.0,
    units=u.s,
    frozen = False,
    convert_tcb2tdb=False)
FD1_Component = 'FD'

FD2 = p.prefixParameter(parameter_type="float",
    name="FD2",
    value=0.0,
    units=u.s,
    frozen = False,
    convert_tcb2tdb=False)
FD2_Component = 'FD'

FD3 = p.prefixParameter(parameter_type="float",
    name="FD3",
    value=0.0,
    units=u.s,
    frozen = False,
    convert_tcb2tdb=False)
FD3_Component = 'FD'

FD4 = p.prefixParameter(parameter_type="float",
    name="FD4",
    value=0.0,
    units=u.s,
    frozen = False,
    convert_tcb2tdb=False)
FD4_Component = 'FD'

FD5 = p.prefixParameter(parameter_type="float",
    name="FD5",
    value=0.0,
    units=u.s,
    frozen = False,
    convert_tcb2tdb=False)
FD5_Component = 'FD'

FD6 = p.prefixParameter(parameter_type="float",
    name="FD6",
    value=0.0,
    units=u.s,
    frozen = False,
    convert_tcb2tdb=False)
FD6_Component = 'FD'

# FX, spindown derivatives
F1 = p.prefixParameter(parameter_type="float",
    name="F1",
    value=0.0,
    units=u.Hz/u.s,
    frozen = False,
    convert_tcb2tdb=False)
F1_Component = 'Spindown'

F2 = p.prefixParameter(parameter_type="float",
    name="F2",
    value=0.0,
    units=u.Hz/u.s/u.s,
    frozen = False,
    convert_tcb2tdb=False)
F2_Component = 'Spindown'

F3 = p.prefixParameter(parameter_type="float",
    name="F3",
    value=0.0,
    units=u.Hz/u.s/u.s/u.s,
    frozen = False,
    convert_tcb2tdb=False)
F3_Component = 'Spindown'
