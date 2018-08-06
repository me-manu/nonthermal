"""
Functions for synchroton radiation
"""

# --- Imports ------ #
import numpy as np
from scipy.integrate import simps
from scipy.special import zeta, gamma
from astropy import constants as c 
from astropy import units as u
# ------------------ #


def dgamma_dt_sync(gamma, B, pitch = 'iso'):
    """
    Calculates the electron energy loss 
    for Synchrotron radiation

    Parameters
    ----------
    gamma: `~numpy.ndarray`
        gamma factor of electrons, n-dim
    B: float
        B field in G
    pitch: float or str, optional
        if str and == 'iso' assume isotropic pitch angle
        otherwise pitch angle in degrees

    Returns
    -------
    -dgamma / dt in 1. / s as a `~astropy.Quantity`
    """
    result = gamma * gamma / (c.m_e.cgs * c.c.cgs * c.c.cgs)
    result *= 1. / 4. / np.pi * (c.c.cgs * c.sigma_T.cgs)
    result *= (B * u.G)**2.
    if type(pitch) == str:
        if pitch == 'iso':
            sin2pitch = 2. / 3.
        else:
            raise ValueError("Pitch Angle value not understood")
    elif type(pitch) == float:
        sin2pitch = np.sin(np.radian(pitch))**2.
    else:
        raise ValueError("Pitch angle must be str of float")
    result *= sin2pitch
    if result.unit == "cm G2 s / g":
        result = result.value / u.s
        print result.unit
    else:
        raise Exception
    return result

def Esync_peak(gamma, B, pitch = 'iso'):
    """
    Calculates the peak frequency of the
    synchrotron spectrum emitted by an electron

    Parameters
    ----------
    gamma: `~numpy.ndarray`
        gamma factor of electrons, n-dim
    B: float
        B field in G
    pitch: float or str, optional
        if str and == 'iso' assume isotropic pitch angle
        otherwise pitch angle in degrees

    Returns
    -------
    peak energy in eV as `~astropy.Quantity`
    """
    result = 3. * c.e.gauss * B * u.G * gamma**2. / 4. \
        / np.pi / c.m_e.cgs / c.c.cgs * c.h.cgs.to('eV s')
    if type(pitch) == str:
        if pitch == 'iso':
            sinpitch = np.sqrt(2. / 3.)
        else:
            raise ValueError("Pitch Angle value not understood")
    elif type(pitch) == float:
        sinpitch = np.sin(np.radian(pitch))
    else:
        raise ValueError("Pitch angle must be str of float")
    result *= sinpitch
    if result.unit == "eV Fr G s2 / (cm g)":
        result = result.value * u.eV
    else:
        raise Exception
    return result
