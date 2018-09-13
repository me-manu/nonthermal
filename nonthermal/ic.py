"""
Functions for inverse Compton scattering
"""

# --- Imports ------ #
import numpy as np
from scipy.integrate import simps
from scipy.special import zeta, gamma
from astropy import constants as c 
from astropy import units as u
# ------------------ #

def phot_dens_black_body(eps, T = 2.726):
    """
    Calculate photon density of black body spectrum 
    in units of 1 / eV / cm^3

    Parameters
    ----------
    eps: `~numpy.ndarray`
        Photon energy in eV
    T: float
        Temperature of black body spectrum in K, optional

    Returns
    -------
    `~astropy.Quantity` with photon densities 
    See e.g. Blumenthal & Gould 1970 Eq. 2.58
    """
    result = (eps * u.eV) ** 2. / (c.hbar * c.c).to('eV cm')**3. / np.pi**2.
    result /= (np.exp(eps * u.eV / (c.k_B * T * u.K).to('eV')) - 1.)
    return result

def integrate_black_body(T = 2.726, s = 4):
    """
    Integrate black body over energy,
    multiply with arbritary powers of energy.

    Parameters
    ----------
    T: float
        Temperature of black body spectrum in K, optional
    s: integer
        power of energy in numerator, for s = 4 calculates integral over photon energy density
        so that average energy is integrate_black_body(s = 4) / integrate_black_body(s = 3)

    Returns
    -------
    `~astropy.Quantity` with total energy density

    Notes
    -----
    Gamma(s) Zeta(s) = int_0^\infty x^{s - 1} / (exp(x) - 1)
    See e.g. https://people.physics.tamu.edu/krisciunas/planck.pdf
    """
    result = (c.k_B * T * u.K).to('eV') ** s
    result /= ((c.hbar * c.c).to('eV cm')**3. * np.pi**2.)
    result *= zeta(s,1) * gamma(s)
    return result

def ic_kernel_thomson(gamma, e, e1):
    """
    Calculate the full inverse Compton Kernel, unitless

    Parameters
    ----------
    gamma: `~numpy.ndarray`
        gamma factor of electrons, n-dim
    e: `~numpy.ndarray`
        initial photon energy in eV, n-dim
    e1: `~numpy.ndarray`
        final photon energy in eV, n-dim

    Returns
    -------
    Full IC kernel as n-dim `~numpy.ndarray`

    Notes
    -----
    see Blumenthal & Gould 1970, Eq. 2.47 - 2.51
    """
    q = e1 / 4. / gamma / gamma / e

    f = np.zeros(q.shape)
    m = (q >= 0.) & (q <= 1.)

    f[m] = 2. * e1[m] * np.log(q[m]) + e1[m] + \
        4. * gamma[m] * gamma[m] * e[m] - \
        e1[m] * e1[m] / 2. /gamma[m] / gamma[m] / e[m]
    return f, m

def ic_scattered_phot_spec_thomson(gamma, e, e1, phot_dens):
    """
    Calculate the scattered photon spectrum dN / dt / de1 /de
    of inverse compton scattering in units 1. / s / eV / eV
    in the thomson regime

    Parameters
    ----------
    gamma: `~numpy.ndarray`
        gamma factor of electrons, n-dim
    e: `~numpy.ndarray`
        initial photon energy in eV, n-dim
    e1: `~numpy.ndarray`
        final photon energy in eV, n-dim
    phot_dens: function
        function that returns photon density in units 1 / eV / cm^3

    Returns
    -------
    Full IC kernel as n-dim `~numpy.ndarray`

    Notes
    -----
    see Blumenthal & Gould 1970, Eq. 2.47 - 2.51
    """
    result, mask = ic_kernel_thomson(gamma, e, e1)
    result *= u.eV
    result *= phot_dens(e) / (e * u.eV) / (e * u.eV)
    result *= 3. / 16. * (c.sigma_T * c.c).to('cm3 s-1') / gamma / gamma / gamma / gamma 
    return result, mask

def ic_kernel(gamma, e, e1):
    """
    Calculate the full inverse Compton Kernel, unitless

    Parameters
    ----------
    gamma: `~numpy.ndarray`
        gamma factor of electrons, n-dim
    e: `~numpy.ndarray`
        initial photon energy in eV, n-dim
    e1: `~numpy.ndarray`
        final photon energy in eV, n-dim

    Returns
    -------
    Full IC kernel as n-dim `~numpy.ndarray`

    Notes
    -----
    see Blumenthal & Gould 1970, Eq. 2.47 - 2.51
    """
    E1 = e1 * u.eV / gamma / (c.m_e * c.c * c.c).to('eV') 
    Gamma = 4. * e * gamma * u.eV/ (c.m_e * c.c * c.c).to('eV')
    q = E1 / Gamma / (1. - E1)

    f = np.zeros(q.shape)

    m = (q >= 1. / 4. / gamma / gamma) & (q <= 1.)
    f[m] = 2. * q[m] * np.log(q[m]) + (1. + 2.*q[m]) * (1. - q[m]) + \
        Gamma[m] * Gamma[m] * q[m] * q[m] / \
        (1. + Gamma[m] * q[m]) / 2. * (1. - q[m])
    return f, m

def ic_scattered_phot_spec(gamma, e, e1, phot_dens):
    """
    Calculate the scattered photon spectrum dN / dt / de1 /de
    of inverse compton scattering in units 1. / s / eV / eV

    Parameters
    ----------
    gamma: `~numpy.ndarray`
        gamma factor of electrons, n-dim
    e: `~numpy.ndarray`
        initial photon energy in eV, n-dim
    e1: `~numpy.ndarray`
        final photon energy in eV, n-dim
    phot_dens: function
        function that returns photon density in units 1 / eV / cm^3

    Returns
    -------
    Full IC kernel as n-dim `~numpy.ndarray`

    Notes
    -----
    see Blumenthal & Gould 1970, Eq. 2.47 - 2.51
    """
    result, mask = ic_kernel(gamma, e, e1)
    result = result * u.dimensionless_unscaled
    result *= phot_dens(e) / (e * u.eV)
    result *= 3. / 4. * (c.sigma_T * c.c).to('cm3 s-1') / gamma / gamma
    return result, mask

def dgamma_dt_Thomson(gamma, phot_dens_iso):
    """
    Calculates the electron energy loss 
    for Thomson scattering on an isotropic photon field

    Parameters
    ----------
    gamma: `~numpy.ndarray`
        gamma factor of electrons, n-dim
    phot_dens_iso: float
        integrated isotropic photon density in eV / cm^3

    Returns
    -------
    -dgamma / dt in 1. / s
    """
    result = gamma * gamma / (c.m_e * c.c * c.c).to('eV')
    result *= 4. / 3. * (c.c * c.sigma_T).to('cm3 s-1')
    result *= phot_dens_iso * u.eV / u.cm**3.
    return result

def tcool_Thomson(gamma, uphot):
    """
    Calculate cooling time in Thomson regime 
    on a photon field.

    Parameters
    ----------
    gamma: float
        Gamma factor of scattered electrons 

    uphot_prime: float
        Energy density of external radiation field in comoving frame
	in erg / cm^3

    Returns
    -------
    float with cooling time in seconds.


    Notes
    -----
    Derivation:
    Starts from |gamma dot| = k gamma ^ 2
    so that 1 / gamma_final - 1 / gamma_initial = k t
    and with gamma_final = 1 / 2 gamma_inital 
    we obtain tcool = 1 / (k gamma).
    """
    if not type(uphot) == u.Quantity:
        uphot = uphot * u.Unit('g cm^-1  s^-2') # erg / cm^3
    # this is simply the factor from dgamma / dt
    result = 3. * c.m_e.cgs * c.c.cgs / c.sigma_T.cgs / uphot / 4.
    # divide by gamma factor from mean scattered energy 
    result /= gamma
    return result

def dgamma_dt_KN(gamma, T = 2.726):
    """
    Calculates the electron energy loss 
    for IC scattering on black body spectrum
    in extreme KN regime

    Parameters
    ----------
    gamma: `~numpy.ndarray`
        gamma factor of electrons, n-dim
    T: float
        Temperature of black body spectrum in K, optional

    Returns
    -------
    -dgamma / dt in 1. / s

    Notes
    -----
    See Blumenthal & Gould Eq. 2.59
    """
    C_E = 0.5772156649
    C_l = 0.5700
    result = np.log(4. * gamma * c.k_B * T * u.K / (c.m_e * c.c ** 2.)) - 5. / 6. 
    result -= (C_E + C_l)
    result *= 1. / 16. * c.sigma_T *c.m_e * (c.k_B * T * u.K) ** 2. / c.hbar ** 3.
    return result


def dgamma_dt_IC(gamma, phot_dens,
        emin = 1e-10, emax = 1e0,
        e1min = None, e1max = None,
	e0 = None,
        esteps = 21, e1steps = 2001):
    """
    Calculate the electron energy loss 
    for generic case of scattering on a photon field
    in units 1. / s

    Parameters
    ----------
    gamma: `~numpy.ndarray`
        gamma factor of electrons, n-dim
    phot_dens: function
        function that returns photon density in units 1 / eV / cm^3
    e0: float or None (optional)
        if float, photon density is treated as being a delta function Delta(e - e0),
    	where e0 is the energy in eV. Therefore it should be an energy integrated 
        photon density in units of 1 / cm^3.
        If None, full integration over photon density is performed

    Returns
    -------
    -dgamma / dt in 1. / s

    Notes
    -----
    see Blumenthal & Gould 1970, Eq. 2.56 in discussion below
    """
    if e0 is None:
        earray = np.logspace(np.log10(emin), 
                np.log10(emax), esteps)
    else:
        earray = np.array([e0])
    if e1max is None:
        e1max = 4. * gamma.max()**2. * 1e-2
    if e1min is None:
        e1min = 10. * emax 
    e1array = np.logspace(np.log10(e1min), 
                np.log10(e1max), e1steps)

    ggg,eee,e111 = np.meshgrid(gamma, earray, e1array, indexing = 'ij')
    dndtdede1, mask = ic_scattered_phot_spec(ggg, eee, e111, phot_dens)

    unit = dndtdede1.unit

    # perform full integration
    gg,ee = np.meshgrid(gamma, earray, indexing = 'ij')
    if e0 is None:
        result = simps(
		    simps(dndtdede1.value * (e111 - eee) * e111, np.log(e111), axis = 2 ) * ee,
		    np.log(ee), axis = 1 )
        return result * unit * u.eV ** 3. / (c.m_e * c.c**2.).to('eV')

    # integration over initial photon energy is delta function
    else:
        result = simps(dndtdede1.value * (e111 - eee) * e111, np.log(e111), axis = 2 )
        result = result[:,0]
        # divide by u.eV due to delta function: delta(e - e0) = delta((e - e0)/eV) / eV
        return result * unit * u.eV ** 3. / (c.m_e * c.c**2.).to('eV') / u.eV
#    result = np.zeros_like(gamma)
#    eee,e111 = np.meshgrid(earray, e1array, indexing = 'ij')
#    for i, g in enumerate(gamma):
#        dndtdede1, mask = ic_scattered_phot_spec(g, eee, e111, phot_dens)
#        unit = dndtdede1.unit
#        result[i] = simps( simps(dndtdede1.value * (eee - e111) * e111, np.log(e111), axis = 1) * earray, 
#                                np.log(earray))


def dN_dt_IC(gamma, phot_dens,
        emin = 1e-10, emax = 1e0,
        e1min = None, e1max = None,
        esteps = 21, e1steps = 2001):
    """
    Calculate the IC spectrum integrated over 
    the scattered photon energy and photon energy
    of the photon field,
    in units 1. / s

    Parameters
    ----------
    gamma: `~numpy.ndarray`
        gamma factor of electrons, n-dim
    phot_dens: function
        function that returns photon density in units 1 / eV / cm^3

    Returns
    -------
    -dgamma / dt in 1. / s

    Notes
    -----
    see Blumenthal & Gould 1970, Eq. 2.56 in discussion below
    """
    earray = np.logspace(np.log10(emin), 
                np.log10(emax), esteps)
    if type(e1max) == type(None):
        e1max = 4. * gamma.max()**2. * 1e-2
    if type(e1min) == type(None):
        e1min = 10. * emax 
    e1array = np.logspace(np.log10(e1min), 
                np.log10(e1max), e1steps)

    ggg,eee,e111 = np.meshgrid(gamma, earray, e1array, indexing = 'ij')
    dndtdede1, mask = ic_scattered_phot_spec(ggg, eee, e111, phot_dens)

    unit = dndtdede1.unit
    gg,ee = np.meshgrid(gamma, earray, indexing = 'ij')
    result = simps( simps(dndtdede1.value * e111, np.log(e111), axis = 2 ) * ee, np.log(ee), axis = 1 )

#    result = np.zeros_like(gamma)
#    eee,e111 = np.meshgrid(earray, e1array, indexing = 'ij')
#    for i, g in enumerate(gamma):
#        dndtdede1, mask = ic_scattered_phot_spec(g, eee, e111, phot_dens)
#        unit = dndtdede1.unit
#        result[i] = simps( simps(dndtdede1.value * (eee - e111) * e111, np.log(e111), axis = 1) * earray, 
#                                np.log(earray))

    return result * unit * u.eV ** 2. 
