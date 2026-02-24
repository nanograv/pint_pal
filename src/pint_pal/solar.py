import numpy as np
import inspect
import jax.numpy as jnp
import warnings
from scipy import interpolate

from discovery import const
from discovery import matrix

AU_light_sec = const.AU / const.c  # 1 AU in light seconds
AU_pc = const.AU / const.pc        # 1 AU in parsecs (for DM normalization)

def theta_impact(psr):
    """From enterprise_extensions: use the attributes of an Enterprise
    Pulsar object to calculate the solar impact angle.
    Returns solar impact angle (rad), distance to Earth (R_earth),
    impact distance (b), perpendicular distance (z_earth)."""

    earth = psr.planetssb[:, 2, :3]
    sun = psr.sunssb[:, :3]
    earthsun = earth - sun

    R_earth = np.sqrt(np.einsum('ij,ij->i', earthsun, earthsun))
    Re_cos_theta_impact = np.einsum('ij,ij->i', earthsun, psr.pos_t)

    theta_impact = np.arccos(-Re_cos_theta_impact / R_earth)
    b = np.sqrt(R_earth**2 - Re_cos_theta_impact**2)

    return theta_impact, R_earth, b, -Re_cos_theta_impact

def make_solardm(psr):
    """From enterprise_extensions: calculate DM
    due to 1/r^2 solar wind density model."""

    theta, r_earth, _, _ = theta_impact(psr)
    shape = matrix.jnparray(AU_light_sec * AU_pc / r_earth / np.sinc(1 - theta/np.pi) * 4.148808e3 / psr.freqs**2)

    def solardm(n_earth):
        return n_earth * shape

    return solardm

def _dm_solar_close(n_earth, r_earth):
     return (n_earth * AU_light_sec * AU_pc / r_earth)


def _dm_solar(n_earth, theta, r_earth):
    return ((np.pi - theta) *
            (n_earth * AU_light_sec * AU_pc
            / (r_earth * np.sin(theta))))

def dm_solar(n_earth, theta, r_earth):
    """
    Calculate dispersion measure from a 1/r^2 solar wind density model.

    This function computes the integrated column density of free electrons
    along the line of sight through the solar wind, assuming a spherically
    symmetric 1/r^2 density profile. The calculation uses different approximations
    depending on the solar elongation angle to avoid numerical issues.

    Parameters
    ----------
    n_earth : float or ndarray
        Solar wind proton/electron density at Earth's orbit (cm^-3).
    theta : float or ndarray
        Solar elongation angle between the Sun and the line of sight to the
        pulsar (radians). theta = 0 corresponds to the Sun directly in the
        line of sight, theta = pi/2 is at right angles.
    r_earth : float or ndarray
        Distance from Earth to Sun (light seconds).

    Returns
    -------
    float or ndarray
        Dispersion measure contribution from solar wind (pc cm^-3).

    Notes
    -----
    For small elongation angles (pi - theta < 1e-5), the function uses a
    close approach approximation to avoid numerical instabilities. Otherwise,
    it uses the full integral formula from the 1/r^2 density model.

    References
    ----------
    .. [1] You, X. P., Hobbs, G., Coles, W. A., et al. 2007, MNRAS, 378, 493
           "Dispersion measure variations and their effect on precision pulsar timing"
           https://doi.org/10.1111/j.1365-2966.2007.11617.x
    """
    return matrix.jnp.where(np.pi - theta >= 1e-5,
                    _dm_solar(n_earth, theta, r_earth),
                    _dm_solar_close(n_earth, r_earth))

def fourierbasis_solar_dm(psr,
                        components,
                        T=None):
    """
    Construct a Fourier design matrix for solar wind dispersion measure variations.


    Parameters
    ----------
    psr : :class:`pulsar.Pulsar`
        Discovery Pulsar object containing TOAs, frequencies, and solar system
        ephemeris information.
    components : int
        Number of Fourier components to include in the model.
    T : float, optional
        Total timespan of the data in seconds. If None, will be computed from
        the pulsar's TOAs.

    Returns
    -------
    f : ndarray
        Sampling frequencies for the Fourier components (Hz).
    df : float
        Frequency spacing between components (Hz).
    F : ndarray
        Solar wind DM-variation Fourier design matrix with shape (n_toas, 2*components),
        where each TOA is weighted by the frequency-dependent solar wind DM delays.

    Notes
    -----
    This function is adapted from enterprise_extensions. The design matrix is
    constructed by first obtaining a standard Fourier basis, then scaling each
    TOA by the solar wind DM signature computed from the 1/r^2 solar wind density
    model.

    Examples
    --------
    Create a Gaussian process model for solar wind DM variations using a powerlaw prior:

    >>> from discovery import solar, signals
    >>> # Create a solar wind DM GP with 30 Fourier components
    >>> gp = signals.makegp_fourier(
    ...     psr,
    ...     signals.powerlaw,
    ...     components=30,
    ...     fourierbasis=solar.fourierbasis_solar_dm,
    ...     name='solar_wind_dm'
    ... )
    """

    # Lazy import to avoid circular dependency
    from discovery.signals import fourierbasis

    # get base Fourier design matrix and frequencies
    f, df, fmat = fourierbasis(psr, components, T)
    theta, R_earth, _, _ = theta_impact(psr)
    dm_sol_wind = dm_solar(1.0, theta, R_earth)
    dt_DM = dm_sol_wind * 4.148808e3 / (psr.freqs**2)

    return f, df, fmat * dt_DM[:, None]

def makegp_timedomain_solar_dm(psr, covariance, dt=1.0, Umat=None, common=[], name='timedomain_sw_gp'):
    """
    Construct a time-domain Gaussian process for solar wind dispersion measure variations.

    This function builds a GP model for solar wind-induced DM variations by combining
    a covariance function in the time domain with a model for the solar wind geometry.
    The TOAs are quantized into time bins, and the GP is constructed using the time separations
    between bins weighted by the solar wind DM signature.

    Parameters
    ----------
    psr : :class:`pulsar.Pulsar`
        Discovery Pulsar object containing TOAs, frequencies, and solar system
        ephemeris information.
    covariance : callable
        Function that returns the time domain autocorrelation for a given
        separation (tau). Must have signature `covariance(tau, *params)` where
        tau is the time separation array.
    dt : float, optional
        Time bin width in seconds for quantizing TOAs. Default is 1.0.
    Umat : ndarray, optional
        Design matrix mapping the low-rank GP to the TOA residuals. If None,
        it will be constructed by quantizing the TOAs and weighting by the solar wind DM signature.
        Default is None.
    common : list, optional
        List of parameter names that should be treated as common (shared) across
        pulsars rather than pulsar-specific. Default is [].
    name : str, optional
        Base name for the GP parameters. Used as prefix for parameter naming.
        Default is 'timedomain_sw_gp'.

    Returns
    -------
    :class:`matrix.VariableGP`
        A matrix.VariableGP object containing the noise covariance matrix (as a
        NoiseMatrix2D_var) and the design matrix (Umat) that maps the GP
        to the TOA residuals via solar wind DM delays. See :class:`matrix.VariableGP`
        for details.

    Notes
    -----
    The design matrix Umat maps the low-rank GP (evaluated at quantized TOAs)
    to the full TOA residuals, scaled by the frequency-dependent solar wind
    DM signature.
    """
    # Lazy import to avoid circular dependency
    from discovery.signals import quantize

    argspec = inspect.getfullargspec(covariance)
    argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}')
              for arg in argspec.args if arg not in ['tau']]

    # get solar wind ingredients
    theta, R_earth, _, _ = theta_impact(psr)
    dm_sol_wind = dm_solar(1.0, theta, R_earth)
    dt_DM = dm_sol_wind * 4.148808e3 / (psr.freqs**2)

    if Umat is None:
        bins = quantize(psr.toas, dt)
        Umat = np.vstack([bins == i for i in range(bins.max() + 1)]).T.astype('d')
        Umat = Umat * dt_DM[:, None]
    else:
        Umat = Umat * dt_DM[:, None]
    toas = psr.toas @ Umat / Umat.sum(axis=0)
    ## i am not fully convinced that this is correct yet.
    ## seems like Daniel is using an Ecorr basis and then doing the averaging.

    get_tmat = covariance
    tau = jnp.abs(toas[:, jnp.newaxis] - toas[jnp.newaxis, :])

    def getphi(params):
        return get_tmat(tau, *[params[arg] for arg in argmap])
    getphi.params = argmap

    return matrix.VariableGP(matrix.NoiseMatrix2D_var(getphi), Umat)


def linear_blocked_sw_interpolation_basis(
        toas,
        bin_edges,
):
    ### this is the basis which Mercedes and I co-wrote
    bin_edges = bin_edges *86400 # MJD to seconds
    ### uses an input of BB then uses solar wind geometry to weight by solar conjunction and dispersion effects
    M = np.zeros((len(toas), len(bin_edges)))
    # make linear interpolation basis
    for ii in range(len(bin_edges) - 1):
        idx = np.logical_and(toas >= bin_edges[ii], toas <= bin_edges[ii + 1])
        M[idx, ii] = (toas[idx] - bin_edges[ii + 1]) / (bin_edges[ii] - bin_edges[ii + 1])
        M[idx, ii + 1] = (toas[idx] - bin_edges[ii]) / (bin_edges[ii + 1] - bin_edges[ii])

    # only return non-zero columns for rank reduction
    idx = M.sum(axis=0) != 0
    
    return M[:, idx], bin_edges[idx]


def custom_blocked_interpolation_basis(
        toas,
        nodes,
        kind="linear",
):
    nodes = nodes * 86400  # MJD to seconds
    basis = np.identity(len(nodes))
    interp = interpolate.interp1d(
        nodes,
        basis,
        kind=kind,
        axis=0,
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )
    M = interp(toas)
    # only return non-zero columns for rank reduction
    idx = M.sum(axis=0) != 0
    if not np.any(idx):
        raise RuntimeError(
            "Interpolation basis has no support in the TOA range. Perhaps check units."
        )

    return M[:, idx], nodes[idx]

def ridge_kernel(
        nodes,
        log10_sigma_ridge: float = -7.
):
    """Ridge regression kernel"""
    def kernel(tau, log10_sigma_ridge=log10_sigma_ridge):
        scale = 10**(2 * log10_sigma_ridge)
        return scale * jnp.eye(nodes, dtype=tau.dtype)

    return kernel

def square_exponential_kernel(
        log10_sigma_se: float = -7.,
        log10_ell_se: float = 1.,
):
    """Square exponential kernel"""
    def kernel(tau, log10_sigma_se=log10_sigma_se, log10_ell_se=log10_ell_se):
        sigma2 = 10**(2 * log10_sigma_se)
        ell = 10**log10_ell_se
        return sigma2 * jnp.exp(-0.5 * (tau / ell)**2)

    return kernel

def quasi_periodic_kernel(
        log10_sigma: float = -7.,
        log10_ell: float = 1.,
        log10_gamma_p: float = 0.,
        log10_p: float = 0.,
):
    """Quasi-periodic kernel"""
    def kernel(tau, log10_sigma=log10_sigma, log10_ell=log10_ell,
               log10_gamma_p=log10_gamma_p, log10_p=log10_p):
        sigma2 = 10**(2 * log10_sigma   )
        ell = 10**log10_ell
        gamma_p = 10**log10_gamma_p
        p = 10**log10_p
        return sigma2 * jnp.exp(-0.5 * (tau / ell)**2 - 2 * (jnp.sin(jnp.pi * tau / p) / gamma_p)**2)

    return kernel



