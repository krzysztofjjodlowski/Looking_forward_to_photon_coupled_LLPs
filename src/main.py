#!/usr/bin/env python
# coding: utf-8

## pyright: reportGeneralTypeIssues=false
## pyright: reportUnusedVariable=warning
## pyright: reportUnboundVariable=false

from numba import njit
import numpy as np

from scipy.integrate import quad  # , simps, nquad, dblquad
from scipy.interpolate import interp1d

from constants import *
from utilities import *


def prob_decay(x_min, x_max, d, isStable=False):
    if isStable: return 0.0
    prob = np.exp(-x_min/d) - np.exp(-x_max/d)
    if (np.isnan(prob) or prob<0): prob = 0
    elif prob > 1: prob = 1
    return prob

def prob_prod_fasernu2_decay_faser2(x_min, Delta, t_max, d, isStable=False):
    """Integrated quasi probability (normed by L_int_inverse) that LLP will decay inside the faser2 detector

    Args:
        x_min (float): minimum distance that LLP has to travel before decay (NOT distance from top of fasernu2 to top of fasernu!)
        Delta (float): length of decay vessel; x_max=x_min+Delta
        t_max (float): length of fasernu2
        d (float): LLP decay length in lab

    Returns:
        float: 
    """

    if isStable: return 0.0
    # if d > 1e10 * x_min:
    #     prob = Delta * t_max / d
    # elif d < 1e10 * x_min:
    #     prob = d * np.exp((t_max - x_min) / d)
    # else:
    #     prob = d * np.exp(-(Delta + x_min) / d) * (np.exp(Delta / d) - 1) * (np.exp(t_max / d) - 1)
    prob = d * np.exp(-(Delta + x_min) / d) * (np.exp(Delta / d) - 1) * (np.exp(t_max / d) - 1)
    if (np.isnan(prob) or prob<0): prob = 0
    elif prob > t_max: prob = t_max
    return prob

def prob_prod_fasernu2_decay_fasernu2(x_min, t_max, d, isStable=False):
    """Integrated quasi probability (normed by L_int_inverse) that LLP will decay inside the faser2nu detector but not before distance x_min

    Args:
        x_min (float): minimum distance that LLP has to travel before decay (NOT distance from top of fasernu2 to top of fasernu!)
        t_max (float): length of fasernu2
        d (float): LLP decay length in lab

    Returns:
        float: 
    """

    if isStable: return 0.0
    # if d > 1e10 * x_min:
    #     prob = t_max**2 / 2 / d
    # elif d < 1e10 * x_min:
    #     prob = t_max * np.exp( -x_min / d)
    # else:
    #     prob = t_max * np.exp(-x_min / d) + d * (np.exp(-t_max / d) - 1)
    prob = t_max * np.exp(-x_min / d) + d * (np.exp(-t_max / d) - 1)
    if (np.isnan(prob) or prob<0): prob = 0
    elif prob > t_max: prob = t_max
    return prob

def prob_just_scattering_fasernu2_decay_outside_faser2(x_max, t_max, d, isStable=False):
    """Integrated quasi probability (normed by L_int_inverse) that LLP will decay outside both faser2nu and faser2 detectors (which are distance x_max=7.2m from top of faser2nu detector) or that it will not decay at all

    Args:
        x_max (float): distance from top of faser2nu detector to the end of faser2 (x_max=7.2m)
        t_max (float): length of fasernu2
        d (float): LLP decay length in lab

    Returns:
        float: 
    """

    if isStable: return t_max
    # if d > 1e10 * x_max:
    #     prob = t_max
    # elif d < 1e10 * x_max:
    #     prob = d * np.exp((t_max - x_max) / d)
    # else:
    #     prob = d * np.exp(-x_max / d) * (np.exp(t_max / d) - 1)
    prob = d * np.exp(-x_max / d) * (np.exp(t_max / d) - 1)
    if (np.isnan(prob) or prob<0): prob = 0
    elif prob > t_max: prob = t_max
    return prob



## Functions for mathematica
def ArcCoth(x): 
    """
        https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#Inverse_hyperbolic_cotangent 
    """ 
    if x>1 or x<-1: return 0.5 * np.log((x+1)/(x-1)) 
    return 0 

def Log(x):
    return np.log(x)

def Sqrt(x):
    return np.sqrt(x)



## Masive Spin-2 (G2)
def Gamma_G2_gg(m_G2, g_gg, Lambda):
    return (g_gg**2 * m_G2**3) / (80. * Lambda**2 * np.pi)

def Gamma_G2_ll(m_G2, g_ll, Lambda, m_l=M_ELECTRON):
    """
    Args:
        m_G2 (float): 
        g_ll (float)
        Lambda (float)
        m_l

        coupling_G2_ll = g_ll/Lambda
    Returns:
        float: 
    """

    m0, m1, m2 = m_G2, m_l, m_l
    if m0 < m1 + m2:
        return 0.0

    return (g_ll**2 * np.sqrt(m0**2 - 4 * m1**2) * (3 * m0**4 - 4 * m0**2 * m1**2 - 32 * m1**4)) / (480. * Lambda**2 * m0**2 * np.pi)


## Analytical form of Primakoff scattering - dsigma/dt integrated from -1 to tMax.
# high_en_limit: expand dsigma/dt = (a0 + a1 * t + a2 * t**2 + ...) / t**2; 
# (a0 + a1 * t) / t**2 * FormFactor**2                  terms are integrated from -Infinity to tMax
# (a2 * t**2 + a3 * t**3 + ...) / t**2 * FormFactor**2  terms are integrated from -1 to tMax.
def sigma_gNucleus_G2Nucleus_analyt(g_gg, s, m1, m3, Z=Z_TUNGSTEN, A=A_TUNGSTEN, Lambda=1.0, t_max=-1e-15):
    # gamma(p1) + Nucleus(p2) -> G2(p3) + Nucleus(p4)
    m1 = 0.0
    m2 = m4 = Z * M_PROTON + (A - Z) * M_NEUTRON  # Nucleus mass in GeV

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if t_max > 0 or t_max < -1: return 0.0
    if t_max > -1e-15: t_max = -1e-15

    a, d = 111.0 * Z**(-1.0 / 3.0) / M_ELECTRON, 0.164 * A**(-2.0 / 3.0)
    sigma = g_gg**2 * ALPHA_EM * Z**2 * (np.log(d / (1 / a**2  - t_max)) - 2.) / (2.0 * Lambda**2)
    return 0.0 if sigma<0 else sigma

def sigma_G2Nucleus_gNucleus_analyt(g_gg, s, m1, m3, Z=Z_TUNGSTEN, A=A_TUNGSTEN, Lambda=1.0, t_max=-1e-15):
    # G2(p1) + Nucleus(p2) -> gamma(p3) + Nucleus(p4)
    m3 = 0.0
    m2 = m4 = Z * M_PROTON + (A - Z) * M_NEUTRON  # Nucleus mass in GeV

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if t_max > 0 or t_max < -1: return 0.0
    if t_max > -1e-15: t_max = -1e-15

    a, d = 111.0 * Z**(-1.0 / 3.0) / M_ELECTRON, 0.164 * A**(-2.0 / 3.0)
    sigma = g_gg**2 * ALPHA_EM * Z**2 * (np.log(d / (1 / a**2  - t_max)) - 2.) / (2.0 * Lambda**2) * 2/5
    return 0.0 if sigma<0 else sigma


## Analytical form of electron scattering - with limits: ERmin < E_R < ERmax (t = 2 * m2 * (m2-ER))
def sigma_ge_G2e_analyt(g_gg, s, m1, m3, ERmin, ERmax, t_min, t_max, Lambda=1.0):
    m1 = 0
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    ER_min, ER_max = ERmin, ERmax

    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0

    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    sigma = g_gg**2 * ALPHA_EM * np.log(ER_max / ER_min) / (2.0 * Lambda**2)
    return 0.0 if sigma<0 else sigma

def sigma_G2e_ge_analyt(g_gg, s, m1, m3, ERmin, ERmax, t_min, t_max, Lambda=1.0):
    m3 = 0
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    ER_min, ER_max = ERmin, ERmax
    
    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0

    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    sigma = g_gg**2 * ALPHA_EM * np.log(ER_max / ER_min) / (2.0 * Lambda**2) * 2/5
    return 0.0 if sigma<0 else sigma


