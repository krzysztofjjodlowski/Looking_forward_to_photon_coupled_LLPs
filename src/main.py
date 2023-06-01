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



## Dark ALP
### gprime
def Gamma_gp_ga(g_aggp, m_gp, m_a):
    if m_gp < m_a:
        return 0.0

    return (g_aggp**2 * m_gp**3 * (1 - m_a**2 / m_gp**2)**3 / (96 * np.pi))

def Gamma_gp_all(g_aggp, m_gp, m_a, m_l=M_ELECTRON):
    """  Three body decay of gprime -> axion e+ e- thru off shell photon    """
    m0 = m_gp
    m1 = m2 = m_l
    m3 = m_a

    s12_min_, s12_max_ = (m1 + m2)**2, (m0 - m3)**2

    if m0 < m1 + m2 + m3:
        return 0.0

    integrand = lambda s23, s12: ( E_EM**2 * g_aggp**2 * (2 * m2**4 * s12 + m0**4 * (2 * m2**2 + s12) + 2 * m2**2 * (m3**4 - m3**2 * s12 - 2 * s12 * s23) + s12 * (m3**4 + s12**2 + 2 * s12 * s23 + 2 * s23**2 - 2 * m3**2 * (s12 + s23)) - 2 * m0**2 * (m2**2 * (2 * m3**2 + s12) + s12 * (s12 + s23)))) / (3. * s12**2) / (256 * np.pi**3 * m0**3)

    return custom_dblquad1(integrand, s12_min_, s12_max_,
                          lambda s12: s23_min_s12(s12, m0, m1, m2, m3),
                          lambda s12: s23_max_s12(s12, m0, m1, m2, m3))[0]

def Gamma_gp_ee(m_gp, epsilon, BR=1.0):
    return (BR * epsilon**2 * E_EM**2 / (12 * np.pi) * m_gp * np.sqrt(1 - 4 * M_ELECTRON**2 / m_gp**2))

### a
def Gamma_a_gg(g_agg, m_a):
    return g_agg**2 * m_a**3 / (64 * np.pi)

def Gamma_a_ggp(g_aggp, m_a, m_gp):
    if m_a < m_gp:
        return 0.0
    return (g_aggp**2 * m_a**3 * (1 - m_gp**2 / m_a**2)**3 / (32 * np.pi))

def Gamma_a_gpgp(g_aggp, m_a, m_gp):
    if m_a < 2 * m_gp:
        return 0.0
    return (g_aggp**2 * m_a**3 *
            (1 - 4 * m_gp / m_a**2)**(3.0 / 2.0) / (64 * np.pi))

def Gamma_a_llgp(g_aggp, m_a, m_gp, m_l=M_ELECTRON):
    """ Three body decay of axion -> gprime e+ e- thru off shell photon    """
    m0 = m_a
    m1 = m2 = m_l
    m3 = m_gp

    if m0 < m1 + m2 + m3:
        return 0.0

    s12_min_, s12_max_ = (m1 + m2)**2, (m0 - m3)**2

    # amplitude 3 times larger than Gamma_gp _eea
    integrand = lambda s23, s12: 3 * ( E_EM**2 * g_aggp**2 * (2 * m2**4 * s12 + m0**4 * (2 * m2**2 + s12) + 2 * m2**2 * (m3**4 - m3**2 * s12 - 2 * s12 * s23) + s12 * (m3**4 + s12**2 + 2 * s12 * s23 + 2 * s23**2 - 2 * m3**2 * (s12 + s23)) - 2 * m0**2 * (m2**2 * (2 * m3**2 + s12) + s12 * (s12 + s23)))) / (3. * s12**2) / (256 * np.pi**3 * m0**3)

    return custom_dblquad(integrand, s12_min_, s12_max_,
                          lambda s12: s23_min_s12(s12, m0, m1, m2, m3),
                          lambda s12: s23_max_s12(s12, m0, m1, m2, m3))[0]


def sigma_gNucleus_aNucleus_200404469(g_agg, s, m1, m3, Z=Z_TUNGSTEN, A=A_TUNGSTEN, t_max=0.0):
    # Note 200404469 uses t>0!
    # for atomic form factors
    a, d = ( 111.0 * Z**(-1.0 / 3.0) / M_ELECTRON, 0.164 * A**(-2.0 / 3.0), )  
    sigma = g_agg**2 / 8 * ALPHA_EM * Z**2 * (np.log((d - t_max) / (1 / a**2 - t_max)) - 2.)
    return sigma

def sigma_aNucleus_gNucleus_200404469(g_agg, s, m1, m3, Z=Z_TUNGSTEN, A=A_TUNGSTEN, t_max=0.0):
    return sigma_gNucleus_aNucleus_200404469(g_agg, s, m1, m3, Z=Z_TUNGSTEN, A=A_TUNGSTEN, t_max=0.0) * 2


def sigma_gpNucleus_aNucleus_analyt(g_aggp, s, m1, m3, Z=Z_TUNGSTEN, A=A_TUNGSTEN, t_max=-1e-15):
    """ 
    gprime (p1) + Nucleus (p2) -> a (p3) + Nucleus (p4)
    """
    m2 = m4 = Z * M_PROTON + (A - Z) * M_NEUTRON  # Nucleus mass in GeV

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if t_max > 0 or t_max < -1: return 0.0
    if t_max > -1e-15: t_max = -1e-15

    a, d = 111.0 * Z**(-1.0 / 3.0) / M_ELECTRON, 0.164 * A**(-2.0 / 3.0)
    sigma = g_aggp**2 * ALPHA_EM * Z**2 * (np.log(d / (1 / a**2  - t_max)) - 2.) / (12.0)
    return 0.0 if sigma<0 else sigma

def sigma_aNucleus_gpNucleus_analyt(g_aggp, s, m1, m3, Z=Z_TUNGSTEN, A=A_TUNGSTEN, t_max=-1e-15):
    """ 
    a (p1) + Nucleus (p2) -> gprime (p3) + Nucleus (p4)
    """
    m2 = m4 = Z * M_PROTON + (A - Z) * M_NEUTRON  # Nucleus mass in GeV

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if t_max > 0 or t_max < -1: return 0.0
    if t_max > -1e-15: t_max = -1e-15

    a, d = 111.0 * Z**(-1.0 / 3.0) / M_ELECTRON, 0.164 * A**(-2.0 / 3.0)
    sigma = g_aggp**2 * ALPHA_EM * Z**2 * (np.log(d / (1 / a**2  - t_max)) - 2.) / (12.0) * 3
    return 0.0 if sigma<0 else sigma

def sigma_gpe_ae_analyt(g_aggp, s, m1, m3, ERmin, ERmax, t_min, t_max):
    """ gprime (p1) + e- (p2) -> a (p3) + e- (p4) """
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    ER_min, ER_max = ERmin, ERmax

    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0

    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    sigma = g_aggp**2 * ALPHA_EM * np.log(ER_max / ER_min) / (12.0)
    return 0.0 if sigma<0 else sigma

def sigma_gpe_ae_log(g_aggp, s, m1, m3, ERmin, ERmax, t_min, t_max, isFLARE=False):
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if isFLARE:
        ERmin, ERmax = 30e-3, 1.0
    else:
        ERmin, ERmax = 0.3, 20.0

    ER_min, ER_max = ERmin, ERmax

    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0
    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    log10ER_min, log10ER_max, nlogER = np.log10(ER_min), np.log10(ER_max), 250
    dlog10ER = (log10ER_max - log10ER_min) / float(nlogER)
    sigma = 0

    # df  = df / dx * dx = df/dx * dlog10x * x * log10
    for log10ER in np.linspace(log10ER_min, log10ER_max, nlogER):
        ER = 10**log10ER  # ER=E4 is energy recoil of target

        t = 2 * m2 * (m2 - ER)
        u = m1**2 + m2**2 + m3**2 + m4**2 - s - t

        E3 = (m2**2 + m3**2 - u) / (2 * m2)
        p3 = np.sqrt(E3**2 - m3**2)
        pR = np.sqrt(ER**2 - m4**2)

        angle = np.arccos((E3 * ER - 0.5 * (s - m3**2 - m4**2)) / p3 / pR)

        if ER > 10:
            cuts = (10 * 1e-3 < angle < 20 * 1e-3)
        elif 3 < ER < 10:
            cuts = (10 * 1e-3 < angle < 30 * 1e-3)
        elif ER < 3:
            cuts = (10 * 1e-3 < angle)

        if isFLARE: cuts = True

        if not cuts: continue

        amp2 = (-2*ALPHAEM*g_aggp**2*Pi*(2*m2**4*t + m1**4*(2*m2**2 + t) + 2*m2**2*(m3**4 - (m3**2 + 2*s)*t) + t*(m3**4 + 2*s**2 + 2*s*t + t**2 - 2*m3**2*(s + t)) - 2*m1**2*(m2**2*(2*m3**2 + t) + t*(s + t)))) / (3.*t**2)

        dsigmadt = amp2 / (16. * np.pi * (m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s)))

        sigma += 2 * m2 * ER * dsigmadt

    sigma *= dlog10ER * np.log(10)
    return 0.0 if sigma<0 else sigma

def sigma_ae_gpe_analyt(g_aggp, s, m1, m3, ERmin, ERmax, t_min, t_max):
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    ER_min, ER_max = ERmin, ERmax
    
    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0

    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    sigma = g_aggp**2 * ALPHA_EM * np.log(ER_max / ER_min) / (12.0) * 3
    return 0.0 if sigma<0 else sigma

def sigma_ae_gpe_log(g_aggp, s, m1, m3, ERmin, ERmax, t_min, t_max, isFLARE=False):
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if isFLARE:
        ERmin, ERmax = 30e-3, 1.0
    else:
        ERmin, ERmax = 0.3, 20.0

    ER_min, ER_max = ERmin, ERmax

    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0
    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    log10ER_min, log10ER_max, nlogER = np.log10(ER_min), np.log10(ER_max), 250
    dlog10ER = (log10ER_max - log10ER_min) / float(nlogER)
    sigma = 0

    # df  = df / dx * dx = df/dx * dlog10x * x * log10
    for log10ER in np.linspace(log10ER_min, log10ER_max, nlogER):
        ER = 10**log10ER  # ER=E4 is energy recoil of target

        t = 2 * m2 * (m2 - ER)
        u = m1**2 + m2**2 + m3**2 + m4**2 - s - t

        E3 = (m2**2 + m3**2 - u) / (2 * m2)
        p3 = np.sqrt(E3**2 - m3**2)
        pR = np.sqrt(ER**2 - m4**2)

        angle = np.arccos((E3 * ER - 0.5 * (s - m3**2 - m4**2)) / p3 / pR)

        if ER > 10:
            cuts = (10 * 1e-3 < angle < 20 * 1e-3)
        elif 3 < ER < 10:
            cuts = (10 * 1e-3 < angle < 30 * 1e-3)
        elif ER < 3:
            cuts = (10 * 1e-3 < angle)

        if isFLARE: cuts = True

        if not cuts: continue

        amp2 = (-2*ALPHAEM*g_aggp**2*Pi*(2*m2**4*t + m1**4*(2*m2**2 + t) + 2*m2**2*(m3**4 - (m3**2 + 2*s)*t) + t*(m3**4 + 2*s**2 + 2*s*t + t**2 - 2*m3**2*(s + t)) - 2*m1**2*(m2**2*(2*m3**2 + t) + t*(s + t))))/t**2

        dsigmadt = amp2 / (16. * np.pi * (m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s)))

        sigma += 2 * m2 * ER * dsigmadt

    sigma *= dlog10ER * np.log(10)
    return 0.0 if sigma<0 else sigma




## magnetic dipole DM
def mag_dip_Gamma_chi1_chi0g(m_chi1, m_chi0, LambdaM=1.0):
    m0, m1, m2 = m_chi1, m_chi0, 0
    if m0 < m1 + m2:
        return 0.0

    mchi0, mchi1 = m_chi0, m_chi1
    return (-mchi0**2 + mchi1**2)**3/(2.*LambdaM**2*mchi1**3*np.pi)

def mag_dip_Gamma_chi1_chi0ll(m_chi1, m_chi0, LambdaM=1.0, m_l=M_ELECTRON):
    m0, m1 = m_chi1, m_chi0
    m2 = m3 = m_l

    if m0 < m1 + m2 + m3:
        return 0.0 

    s12_min_, s12_max_ = (m1 + m2)**2, (m0 - m3)**2

    integrand = lambda s23, s12: (64*ALPHA_EM*np.pi*(2*(m0**2 - m1**2)**2*m2**2 + (m0**4 + m1**4 - 4*m0*m1*m2**2 + 2*(m2**2 - s12)**2 - 2*m0**2*s12 - 2*m1**2*s12)*s23 - ((m0 + m1)**2 + 2*(m2**2 - s12))*s23**2)) / (LambdaM**2*s23**2) / (256 * np.pi**3 * m0**3)
    return custom_dblquad(integrand, s12_min_, s12_max_,
                          lambda s12: s23_min_s12(s12, m0, m1, m2, m3),
                          lambda s12: s23_max_s12(s12, m0, m1, m2, m3))[0]


def mag_dip_sigma_chi0Nucleus_chi1Nucleus_analyt(LambdaM, s, m1, m3, Z=Z_TUNGSTEN, A=A_TUNGSTEN, t_max=-1e-15):
    """ chi0(p1) + Nucleus(p2) -> chi1(p3) + Nucleus(p4) """
    m2 = m4 = Z * M_PROTON + (A - Z) * M_NEUTRON  # Nucleus mass in GeV

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if t_max > 0 or t_max < -1: return 0.0
    if t_max > -1e-15: t_max = -1e-15

    a, d = 111.0 * Z**(-1.0 / 3.0) / M_ELECTRON, 0.164 * A**(-2.0 / 3.0)

    sigma = 4 * ALPHA_EM * Z**2 * (np.log(d / (1 / a**2  - t_max)) - 2.) / (LambdaM**2)
    return 0.0 if sigma<0 else sigma

def mag_dip_sigma_chi0e_chi1e_analyt(LambdaM, s, m1, m3, ERmin, ERmax, t_min, t_max):
    """ chi0 (p1) + e- (p2) -> chi1 (p3) + e- (p4) """
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    ER_min, ER_max = ERmin, ERmax
    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0

    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    sigma = 4 * ALPHA_EM * np.log(ER_max / ER_min) / (LambdaM**2)
    return 0.0 if sigma<0 else sigma

def mag_dip_sigma_chi0e_chi1e_log(LambdaM, s, m1, m3, ERmin, ERmax, t_min, t_max, isFLARE=False):
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if isFLARE:
        ERmin, ERmax = 30e-3, 1.0
    else:
        ERmin, ERmax = 0.3, 20.0

    ER_min, ER_max = ERmin, ERmax

    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0
    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    log10ER_min, log10ER_max, nlogER = np.log10(ER_min), np.log10(ER_max), 250
    dlog10ER = (log10ER_max - log10ER_min) / float(nlogER)
    sigma = 0

    mchi0 = m1
    mchi1 = m3

    # df  = df / dx * dx = df/dx * dlog10x * x * log10
    for log10ER in np.linspace(log10ER_min, log10ER_max, nlogER):
        ER = 10**log10ER  # ER=E4 is energy recoil of target

        t = 2 * m2 * (m2 - ER)
        u = m1**2 + m2**2 + m3**2 + m4**2 - s - t

        E3 = (m2**2 + m3**2 - u) / (2 * m2)
        p3 = np.sqrt(E3**2 - m3**2)
        pR = np.sqrt(ER**2 - m4**2)

        angle = np.arccos((E3 * ER - 0.5 * (s - m3**2 - m4**2)) / p3 / pR)

        if ER > 10:
            cuts = (10 * 1e-3 < angle < 20 * 1e-3)
        elif 3 < ER < 10:
            cuts = (10 * 1e-3 < angle < 30 * 1e-3)
        elif ER < 3:
            cuts = (10 * 1e-3 < angle)

        if isFLARE: cuts = True

        if not cuts: continue

        amp2 = (32*ALPHAEM*Pi*(-2*(mchi0**2 - mchi1**2)**2*M_ELECTRON**2 - (mchi0**4 + mchi1**4 - 4*mchi0*mchi1*M_ELECTRON**2 + 2*(M_ELECTRON**2 - s)**2 - 2*mchi0**2*s - 2*mchi1**2*s)*t + ((mchi0 + mchi1)**2 + 2*(M_ELECTRON**2 - s))*t**2))/(LambdaM**2*t**2)

        dsigmadt = amp2 / (16. * np.pi * (m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s)))

        sigma += 2 * m2 * ER * dsigmadt

    sigma *= dlog10ER * np.log(10)
    return 0.0 if sigma<0 else sigma

def mag_dip_sigma_chi1e_chi0e_log(LambdaM, s, m1, m3, ERmin, ERmax, t_min, t_max, isFLARE=False):
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if isFLARE:
        ERmin, ERmax = 30e-3, 1.0
    else:
        ERmin, ERmax = 0.3, 20.0

    ER_min, ER_max = ERmin, ERmax

    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0
    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    log10ER_min, log10ER_max, nlogER = np.log10(ER_min), np.log10(ER_max), 250
    dlog10ER = (log10ER_max - log10ER_min) / float(nlogER)
    sigma = 0

    mchi0 = m1
    mchi1 = m3

    # df  = df / dx * dx = df/dx * dlog10x * x * log10
    for log10ER in np.linspace(log10ER_min, log10ER_max, nlogER):
        ER = 10**log10ER  # ER=E4 is energy recoil of target

        t = 2 * m2 * (m2 - ER)
        u = m1**2 + m2**2 + m3**2 + m4**2 - s - t

        E3 = (m2**2 + m3**2 - u) / (2 * m2)
        p3 = np.sqrt(E3**2 - m3**2)
        pR = np.sqrt(ER**2 - m4**2)

        angle = np.arccos((E3 * ER - 0.5 * (s - m3**2 - m4**2)) / p3 / pR)

        if ER > 10:
            cuts = (10 * 1e-3 < angle < 20 * 1e-3)
        elif 3 < ER < 10:
            cuts = (10 * 1e-3 < angle < 30 * 1e-3)
        elif ER < 3:
            cuts = (10 * 1e-3 < angle)

        if isFLARE: cuts = True

        if not cuts: continue

        amp2 = (32*ALPHAEM*Pi*(-2*(mchi0**2 - mchi1**2)**2*M_ELECTRON**2 - (mchi0**4 + mchi1**4 - 4*mchi0*mchi1*M_ELECTRON**2 + 2*(M_ELECTRON**2 - s)**2 - 2*mchi0**2*s - 2*mchi1**2*s)*t + ((mchi0 + mchi1)**2 + 2*(M_ELECTRON**2 - s))*t**2))/(LambdaM**2*t**2)

        dsigmadt = amp2 / (16. * np.pi * (m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s)))

        sigma += 2 * m2 * ER * dsigmadt

    sigma *= dlog10ER * np.log(10)
    return 0.0 if sigma<0 else sigma




## electric dipole DM
def el_dip_Gamma_chi1_chi0g(m_chi1, m_chi0, LambdaE=1.0):
    m0, m1, m2 = m_chi1, m_chi0, 0
    if m0 < m1 + m2:
        return 0.0

    mchi0, mchi1 = m_chi0, m_chi1
    return (m0**2 - m1**2)**3/(2.*LambdaE**2*m0**3*np.pi)

def el_dip_Gamma_chi1_chi0ll(m_chi1, m_chi0, LambdaE=1.0, m_l=M_ELECTRON):
    m0, m1 = m_chi1, m_chi0
    m2 = m3 = m_l

    if m0 < m1 + m2 + m3:
        return 0.0 

    s12_min_, s12_max_ = (m1 + m2)**2, (m0 - m3)**2

    integrand = lambda s23, s12: (64*ALPHA_EM*np.pi*(2*(m0**2 - m1**2)**2*m2**2 + (m0**4 + m1**4 + 4*m0*m1*m2**2 + 2*(m2**2 - s12)**2 - 2*m0**2*s12 - 2*m1**2*s12)*s23 - ((m0 - m1)**2 + 2*(m2**2 - s12))*s23**2)) / (LambdaE**2*s23**2) / (256 * np.pi**3 * m0**3)   
    return custom_dblquad(integrand, s12_min_, s12_max_,
                          lambda s12: s23_min_s12(s12, m0, m1, m2, m3),
                          lambda s12: s23_max_s12(s12, m0, m1, m2, m3))[0]


def el_dip_sigma_chi0Nucleus_chi1Nucleus_analyt(LambdaE, s, m1, m3, Z=Z_TUNGSTEN, A=A_TUNGSTEN, t_max=-1e-15):
    """ chi0(p1) + Nucleus(p2) -> chi1(p3) + Nucleus(p4) """
    m2 = m4 = Z * M_PROTON + (A - Z) * M_NEUTRON  # Nucleus mass in GeV

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if t_max > 0 or t_max < -1: return 0.0
    if t_max > -1e-15: t_max = -1e-15

    a, d = 111.0 * Z**(-1.0 / 3.0) / M_ELECTRON, 0.164 * A**(-2.0 / 3.0)

    sigma = 4 * ALPHA_EM * Z**2 * (np.log(d / (1 / a**2  - t_max)) - 2.) / (LambdaE**2)
    return 0.0 if sigma<0 else sigma

def el_dip_sigma_chi0e_chi1e_analyt(LambdaE, s, m1, m3, ERmin, ERmax, t_min, t_max):
    """ chi0 (p1) + e- (p2) -> chi1 (p3) + e- (p4) """
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    ER_min, ER_max = ERmin, ERmax
    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0

    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    sigma = 4 * ALPHA_EM * np.log(ER_max / ER_min) / (LambdaE**2)
    return 0.0 if sigma<0 else sigma

def el_dip_sigma_chi0e_chi1e_log(LambdaE, s, m1, m3, ERmin, ERmax, t_min, t_max, isFLARE=False):
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if isFLARE:
        ERmin, ERmax = 30e-3, 1.0
    else:
        ERmin, ERmax = 0.3, 20.0

    ER_min, ER_max = ERmin, ERmax

    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0
    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    log10ER_min, log10ER_max, nlogER = np.log10(ER_min), np.log10(ER_max), 250
    dlog10ER = (log10ER_max - log10ER_min) / float(nlogER)
    sigma = 0

    mchi0 = m1
    mchi1 = m3

    # df  = df / dx * dx = df/dx * dlog10x * x * log10
    for log10ER in np.linspace(log10ER_min, log10ER_max, nlogER):
        ER = 10**log10ER  # ER=E4 is energy recoil of target

        t = 2 * m2 * (m2 - ER)
        u = m1**2 + m2**2 + m3**2 + m4**2 - s - t

        E3 = (m2**2 + m3**2 - u) / (2 * m2)
        p3 = np.sqrt(E3**2 - m3**2)
        pR = np.sqrt(ER**2 - m4**2)

        angle = np.arccos((E3 * ER - 0.5 * (s - m3**2 - m4**2)) / p3 / pR)

        if ER > 10:
            cuts = (10 * 1e-3 < angle < 20 * 1e-3)
        elif 3 < ER < 10:
            cuts = (10 * 1e-3 < angle < 30 * 1e-3)
        elif ER < 3:
            cuts = (10 * 1e-3 < angle)

        if isFLARE: cuts = True

        if not cuts: continue

        amp2 = (32*ALPHAEM*Pi*(-2*(mchi0**2 - mchi1**2)**2*M_ELECTRON**2 - (mchi0**4 + mchi1**4 + 4*mchi0*mchi1*M_ELECTRON**2 + 2*(M_ELECTRON**2 - s)**2 - 2*mchi0**2*s - 2*mchi1**2*s)*t + ((mchi0 - mchi1)**2 + 2*(M_ELECTRON**2 - s))*t**2))/(LambdaE**2*t**2)
        
        dsigmadt = amp2 / (16. * np.pi * (m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s)))

        sigma += 2 * m2 * ER * dsigmadt

    sigma *= dlog10ER * np.log(10)
    return 0.0 if sigma<0 else sigma

def el_dip_sigma_chi1e_chi0e_log(LambdaE, s, m1, m3, ERmin, ERmax, t_min, t_max, isFLARE=False):
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if isFLARE:
        ERmin, ERmax = 30e-3, 1.0
    else:
        ERmin, ERmax = 0.3, 20.0

    ER_min, ER_max = ERmin, ERmax

    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0
    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    log10ER_min, log10ER_max, nlogER = np.log10(ER_min), np.log10(ER_max), 250
    dlog10ER = (log10ER_max - log10ER_min) / float(nlogER)
    sigma = 0

    mchi0 = m1
    mchi1 = m3

    # df  = df / dx * dx = df/dx * dlog10x * x * log10
    for log10ER in np.linspace(log10ER_min, log10ER_max, nlogER):
        ER = 10**log10ER  # ER=E4 is energy recoil of target

        t = 2 * m2 * (m2 - ER)
        u = m1**2 + m2**2 + m3**2 + m4**2 - s - t

        E3 = (m2**2 + m3**2 - u) / (2 * m2)
        p3 = np.sqrt(E3**2 - m3**2)
        pR = np.sqrt(ER**2 - m4**2)

        angle = np.arccos((E3 * ER - 0.5 * (s - m3**2 - m4**2)) / p3 / pR)

        if ER > 10:
            cuts = (10 * 1e-3 < angle < 20 * 1e-3)
        elif 3 < ER < 10:
            cuts = (10 * 1e-3 < angle < 30 * 1e-3)
        elif ER < 3:
            cuts = (10 * 1e-3 < angle)

        if isFLARE: cuts = True

        if not cuts: continue

        amp2 = (32*ALPHAEM*Pi*(-2*(mchi0**2 - mchi1**2)**2*M_ELECTRON**2 - (mchi0**4 + mchi1**4 + 4*mchi0*mchi1*M_ELECTRON**2 + 2*(M_ELECTRON**2 - s)**2 - 2*mchi0**2*s - 2*mchi1**2*s)*t + ((mchi0 - mchi1)**2 + 2*(M_ELECTRON**2 - s))*t**2))/(LambdaE**2*t**2)
        
        dsigmadt = amp2 / (16. * np.pi * (m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s)))

        sigma += 2 * m2 * ER * dsigmadt

    sigma *= dlog10ER * np.log(10)
    return 0.0 if sigma<0 else sigma



def Br_Gamma_DP_ee(m_DP):
    tab = np.array([
           [1.00000000e-03, 1.00000000e+00],
           [1.78753170e-01, 1.00000000e+00],
           [1.80223730e-01, 1.00000000e+00],
           [1.81706400e-01, 1.00000000e+00],
           [1.83201250e-01, 1.00000000e+00],
           [1.84708400e-01, 1.00000000e+00],
           [1.86227960e-01, 1.00000000e+00],
           [1.87760030e-01, 1.00000000e+00],
           [1.89304680e-01, 1.00000000e+00],
           [1.90862060e-01, 1.00000000e+00],
           [1.92432220e-01, 1.00000000e+00],
           [1.94015320e-01, 1.00000000e+00],
           [1.95611450e-01, 1.00000000e+00],
           [1.97220700e-01, 1.00000000e+00],
           [1.98843180e-01, 1.00000000e+00],
           [2.00479030e-01, 1.00000000e+00],
           [2.02128320e-01, 1.00000000e+00],
           [2.03791190e-01, 1.00000000e+00],
           [2.05467730e-01, 1.00000000e+00],
           [2.07158070e-01, 1.00000000e+00],
           [2.08862300e-01, 9.99303880e-01],
           [2.10580570e-01, 9.63854490e-01],
           [2.12312970e-01, 9.09243170e-01],
           [2.14059620e-01, 8.45050990e-01],
           [2.15820640e-01, 7.94272070e-01],
           [2.17596160e-01, 7.52115960e-01],
           [2.19386280e-01, 7.22415090e-01],
           [2.21191120e-01, 7.02295120e-01],
           [2.23010790e-01, 6.85049470e-01],
           [2.24845470e-01, 6.68761910e-01],
           [2.26695220e-01, 6.55348600e-01],
           [2.28560190e-01, 6.42893430e-01],
           [2.30440500e-01, 6.33312460e-01],
           [2.32336280e-01, 6.24689640e-01],
           [2.34247670e-01, 6.17024900e-01],
           [2.36174780e-01, 6.10318240e-01],
           [2.38117710e-01, 6.04569670e-01],
           [2.40076660e-01, 5.98821160e-01],
           [2.42051720e-01, 5.94988760e-01],
           [2.44043040e-01, 5.90198280e-01],
           [2.46050700e-01, 5.86365940e-01],
           [2.48074920e-01, 5.82533600e-01],
           [2.50115780e-01, 5.78701140e-01],
           [2.52173420e-01, 5.74868800e-01],
           [2.54247990e-01, 5.71994480e-01],
           [2.56339640e-01, 5.69120290e-01],
           [2.58448480e-01, 5.66245970e-01],
           [2.60574700e-01, 5.63371720e-01],
           [2.62718380e-01, 5.60497400e-01],
           [2.64879700e-01, 5.58581170e-01],
           [2.67058790e-01, 5.55706980e-01],
           [2.69255850e-01, 5.53790750e-01],
           [2.71470960e-01, 5.51874520e-01],
           [2.73704260e-01, 5.49000320e-01],
           [2.75955980e-01, 5.47084090e-01],
           [2.78226200e-01, 5.45167980e-01],
           [2.80515130e-01, 5.43251750e-01],
           [2.82822850e-01, 5.41335520e-01],
           [2.85149570e-01, 5.39419350e-01],
           [2.87495430e-01, 5.37503180e-01],
           [2.89860610e-01, 5.35587010e-01],
           [2.92245210e-01, 5.33670780e-01],
           [2.94649450e-01, 5.31754670e-01],
           [2.97073450e-01, 5.29838440e-01],
           [2.99517420e-01, 5.27922210e-01],
           [3.01981480e-01, 5.26006100e-01],
           [3.04465800e-01, 5.24089870e-01],
           [3.06970600e-01, 5.22173700e-01],
           [3.09495960e-01, 5.20257530e-01],
           [3.12042120e-01, 5.17383220e-01],
           [3.14609200e-01, 5.15467050e-01],
           [3.17197440e-01, 5.13550880e-01],
           [3.19806960e-01, 5.11634710e-01],
           [3.22437940e-01, 5.09718480e-01],
           [3.25090560e-01, 5.07802310e-01],
           [3.27764990e-01, 5.05886140e-01],
           [3.30461440e-01, 5.03969910e-01],
           [3.33180100e-01, 5.02053800e-01],
           [3.35921110e-01, 5.00137570e-01],
           [3.38684620e-01, 4.98221400e-01],
           [3.41470930e-01, 4.96305170e-01],
           [3.44280120e-01, 4.94389000e-01],
           [3.47112450e-01, 4.92472830e-01],
           [3.49968050e-01, 4.89598570e-01],
           [3.52847160e-01, 4.87682340e-01],
           [3.55749960e-01, 4.85766170e-01],
           [3.58676640e-01, 4.83849940e-01],
           [3.61627370e-01, 4.81933830e-01],
           [3.64602420e-01, 4.80017600e-01],
           [3.67601930e-01, 4.78101430e-01],
           [3.70626090e-01, 4.76185260e-01],
           [3.73675140e-01, 4.74269030e-01],
           [3.76749280e-01, 4.72352860e-01],
           [3.79848720e-01, 4.70436690e-01],
           [3.82973670e-01, 4.68520520e-01],
           [3.86124310e-01, 4.66604290e-01],
           [3.89300850e-01, 4.64688180e-01],
           [3.92503530e-01, 4.62771950e-01],
           [3.95732580e-01, 4.60855720e-01],
           [3.98988190e-01, 4.58939550e-01],
           [4.02270590e-01, 4.57023380e-01],
           [4.05579950e-01, 4.55107210e-01],
           [4.08916560e-01, 4.53190980e-01],
           [4.12280650e-01, 4.51274870e-01],
           [4.15672360e-01, 4.48400560e-01],
           [4.19092060e-01, 4.46484390e-01],
           [4.22539800e-01, 4.44568220e-01],
           [4.26015910e-01, 4.42651990e-01],
           [4.29520700e-01, 4.40735820e-01],
           [4.33054240e-01, 4.38819590e-01],
           [4.36616930e-01, 4.36903420e-01],
           [4.40208850e-01, 4.34029160e-01],
           [4.43830340e-01, 4.32112990e-01],
           [4.47481660e-01, 4.30196760e-01],
           [4.51162960e-01, 4.28280590e-01],
           [4.54874580e-01, 4.26364420e-01],
           [4.58616760e-01, 4.24448250e-01],
           [4.62389680e-01, 4.22532080e-01],
           [4.66193680e-01, 4.20615850e-01],
           [4.70028940e-01, 4.18699680e-01],
           [4.73895730e-01, 4.15825430e-01],
           [4.77794410e-01, 4.13909200e-01],
           [4.81725100e-01, 4.11993030e-01],
           [4.85688090e-01, 4.10076860e-01],
           [4.89683810e-01, 4.07202540e-01],
           [4.93712280e-01, 4.05286370e-01],
           [4.97774000e-01, 4.02412120e-01],
           [5.01869020e-01, 4.00495890e-01],
           [5.05997780e-01, 3.96663550e-01],
           [5.10160570e-01, 3.92831150e-01],
           [5.14357510e-01, 3.88998750e-01],
           [5.18589080e-01, 3.85166410e-01],
           [5.22855340e-01, 3.81334070e-01],
           [5.27156710e-01, 3.77501670e-01],
           [5.31493540e-01, 3.73669300e-01],
           [5.35866020e-01, 3.69836960e-01],
           [5.40274440e-01, 3.66004560e-01],
           [5.44719220e-01, 3.62172190e-01],
           [5.49200480e-01, 3.58339820e-01],
           [5.53718690e-01, 3.53549360e-01],
           [5.58273970e-01, 3.49716990e-01],
           [5.62866690e-01, 3.44926540e-01],
           [5.67497310e-01, 3.40136050e-01],
           [5.72165970e-01, 3.35345600e-01],
           [5.76873120e-01, 3.29597060e-01],
           [5.81618910e-01, 3.20974200e-01],
           [5.86403730e-01, 3.13309460e-01],
           [5.91227950e-01, 3.05644690e-01],
           [5.96091870e-01, 2.97021870e-01],
           [6.00995720e-01, 2.88399040e-01],
           [6.05940040e-01, 2.80734300e-01],
           [6.10924960e-01, 2.73069560e-01],
           [6.15950940e-01, 2.63488620e-01],
           [6.21018170e-01, 2.55823880e-01],
           [6.26127120e-01, 2.48159140e-01],
           [6.31278220e-01, 2.40494400e-01],
           [6.36471570e-01, 2.30913460e-01],
           [6.41707600e-01, 2.23248720e-01],
           [6.46986840e-01, 2.15583980e-01],
           [6.52309420e-01, 2.07919240e-01],
           [6.57675920e-01, 2.00254500e-01],
           [6.63086410e-01, 1.91631670e-01],
           [6.68541430e-01, 1.83966930e-01],
           [6.74041450e-01, 1.76302160e-01],
           [6.79586590e-01, 1.70553620e-01],
           [6.85177450e-01, 1.65763140e-01],
           [6.90814260e-01, 1.60972680e-01],
           [6.96497380e-01, 1.57140310e-01],
           [7.02227350e-01, 1.53307940e-01],
           [7.08004360e-01, 1.50433690e-01],
           [7.13828920e-01, 1.47559370e-01],
           [7.19701530e-01, 1.43727030e-01],
           [7.25622300e-01, 1.41810830e-01],
           [7.31591880e-01, 1.37978460e-01],
           [7.37610520e-01, 1.36062260e-01],
           [7.43678630e-01, 1.33188010e-01],
           [7.49796750e-01, 1.30313720e-01],
           [7.55965110e-01, 1.28397520e-01],
           [7.62184320e-01, 1.26481340e-01],
           [7.68454610e-01, 1.19774680e-01],
           [7.74776520e-01, 1.06361400e-01],
           [7.81150460e-01, 7.18700400e-02],
           [7.87576790e-01, 6.32472200e-02],
           [7.94055940e-01, 9.19899900e-02],
           [8.00588550e-01, 1.10193770e-01],
           [8.07174740e-01, 1.19774680e-01],
           [8.13815300e-01, 1.25523250e-01],
           [8.20510330e-01, 1.30313720e-01],
           [8.27260430e-01, 1.36062260e-01],
           [8.34066210e-01, 1.41810830e-01],
           [8.40927840e-01, 1.50433690e-01],
           [8.47846030e-01, 1.58098430e-01],
           [8.54821030e-01, 1.66721250e-01],
           [8.61853360e-01, 1.75344080e-01],
           [8.68943750e-01, 1.84925020e-01],
           [8.76092310e-01, 1.94505930e-01],
           [8.83299650e-01, 2.08877330e-01],
           [8.90566470e-01, 2.28039180e-01],
           [8.97892890e-01, 2.46242940e-01],
           [9.05279760e-01, 2.57740080e-01],
           [9.12727240e-01, 2.63488620e-01],
           [9.20235990e-01, 2.68279080e-01],
           [9.27806620e-01, 2.73069560e-01],
           [9.35439410e-01, 2.78818100e-01],
           [9.43135020e-01, 2.84566670e-01],
           [9.50894060e-01, 2.91273330e-01],
           [9.58716810e-01, 2.97021870e-01],
           [9.66604050e-01, 3.03728520e-01],
           [9.74556030e-01, 3.13309460e-01],
           [9.82573450e-01, 3.23848460e-01],
           [9.90656910e-01, 3.13309460e-01],
           [9.98806770e-01, 2.67321020e-01],
           [1.00702381e+00, 1.93547840e-01],
           [1.01530838e+00, 1.44685120e-01],
           [1.02366102e+00, 1.90673560e-01],
           [1.03208256e+00, 2.50075310e-01],
           [1.04057324e+00, 2.85524760e-01],
           [1.04913366e+00, 2.97021870e-01],
           [1.05776477e+00, 3.08518980e-01],
           [1.06646669e+00, 3.20016120e-01],
           [1.07524037e+00, 3.31513230e-01],
           [1.08408606e+00, 3.43010340e-01],
           [1.09300458e+00, 3.51633160e-01],
           [1.10199654e+00, 3.55465530e-01],
           [1.11106241e+00, 3.55465530e-01],
           [1.12020290e+00, 3.53549360e-01],
           [1.12941849e+00, 3.53549360e-01],
           [1.13870990e+00, 3.53549360e-01],
           [1.14807796e+00, 3.51633160e-01],
           [1.15752292e+00, 3.51633160e-01],
           [1.16704547e+00, 3.49716990e-01],
           [1.17664659e+00, 3.49716990e-01],
           [1.18632662e+00, 3.47800820e-01],
           [1.19608629e+00, 3.45884590e-01],
           [1.20592618e+00, 3.45884590e-01],
           [1.21584702e+00, 3.43968420e-01],
           [1.22584963e+00, 3.42052250e-01],
           [1.23593438e+00, 3.41094170e-01],
           [1.24610198e+00, 3.39177970e-01],
           [1.25635350e+00, 3.38219880e-01],
           [1.26668918e+00, 3.36303680e-01],
           [1.27711010e+00, 3.34387510e-01],
           [1.28761649e+00, 3.32471310e-01],
           [1.29820943e+00, 3.29597060e-01],
           [1.30888963e+00, 3.27680860e-01],
           [1.31965744e+00, 3.23848460e-01],
           [1.33051407e+00, 3.20974200e-01],
           [1.34145987e+00, 3.16183750e-01],
           [1.35249567e+00, 3.10435180e-01],
           [1.36362255e+00, 3.02770440e-01],
           [1.37484086e+00, 2.95105700e-01],
           [1.38615108e+00, 2.92231410e-01],
           [1.39755476e+00, 2.90315210e-01],
           [1.40905225e+00, 2.88399040e-01],
           [1.42064393e+00, 2.86482840e-01],
           [1.43233132e+00, 2.84566670e-01],
           [1.44411492e+00, 2.83608560e-01],
           [1.45599508e+00, 2.82650500e-01],
           [1.46797335e+00, 2.80734300e-01],
           [1.48005021e+00, 2.78818100e-01],
           [1.49222589e+00, 2.76901930e-01],
           [1.50450230e+00, 2.74985730e-01],
           [1.51687956e+00, 2.73069560e-01],
           [1.52935874e+00, 2.71153360e-01],
           [1.54194021e+00, 2.69237190e-01],
           [1.55462551e+00, 2.68279080e-01],
           [1.56741524e+00, 2.66362910e-01],
           [1.58030963e+00, 2.63488620e-01],
           [1.59331059e+00, 2.61572450e-01],
           [1.60641861e+00, 2.56781970e-01],
           [1.61963391e+00, 2.48159140e-01],
           [1.63295841e+00, 2.36662030e-01],
           [1.64639258e+00, 2.23248720e-01],
           [1.65993679e+00, 2.19416350e-01],
           [1.67359281e+00, 2.30913460e-01],
           [1.68736124e+00, 2.38578230e-01],
           [1.80000000e+00, 2.40000000e-01],
           [1.90000000e+00, 2.44000000e-01],
           [2.00000000e+00, 2.48000000e-01],
           [2.10000000e+00, 2.52000000e-01],
           [2.27722132e+00, 2.54756660e-01],
           [2.48116824e+00, 2.46375191e-01],
           [2.67349506e+00, 2.18156653e-01],
           [2.94184425e+00, 2.09015596e-01],
           [3.23598735e+00, 2.03457229e-01],
           [3.52692013e+00, 1.91851339e-01],
           [3.88196962e+00, 1.79991652e-01],
           [4.29773524e+00, 1.75310795e-01],
           [4.72543845e+00, 1.74072732e-01],
           [5.23007617e+00, 1.72236953e-01],
           [5.74921653e+00, 1.73380325e-01],
           [6.36182584e+00, 1.73715235e-01],
           [6.99445046e+00, 1.73193710e-01],
           [7.71424864e+00, 1.73859089e-01],
           [8.51019900e+00, 1.72043682e-01],
           [9.31611537e+00, 1.68952473e-01],
           [1.02890128e+01, 1.68887145e-01],
           [1.14064108e+01, 1.70372551e-01],
           [1.24989134e+01, 1.70543979e-01],
           [1.40152596e+01, 1.73986857e-01],
           [1.45343932e+01, 1.77903363e-01],
           [14.5343932e+01, 1.77903363e-01]
           ])

    BR = interp1d(tab[:,0], tab[:,1], kind='linear', fill_value='extrapolate')(m_DP)
    BR = np.float(BR)
    if BR > 1.:
        BR = 1.
    if BR < 0.:
        print("Br_Gamma_DP_ee(m_DP)=0, m_DP=", m_DP)
        BR = 0.
    return BR

# def Gamma_DP(m_DP, epsilon):
#     # total decay width of dark photon (DP) due to kinetic mixing
#     BR = Br_Gamma_DP_ee(m_DP)
#     return epsilon**2 * ALPHA_EM * m_DP / 3.0 * (1 + 2 * M_ELECTRON**2 / m_DP**2) * np.sqrt(1 - 4 * M_ELECTRON**2 / m_DP**2) / BR

## anapole DM
def anap_Gamma_chi1_chi0ll(m_chi1, m_chi0, Lambdaa=1.0, m_l=M_ELECTRON, is_Br_Gamma_DP_ee=False):
    m0, m1 = m_chi1, m_chi0
    m2 = m3 = m_l

    if m0 < m1 + m2 + m3:
        return 0.0 

    s12_min_, s12_max_ = (m1 + m2)**2, (m0 - m3)**2

    if is_Br_Gamma_DP_ee:
        integrand = lambda s23, s12: 1/Lambdaa**2 * (-16)*ALPHA_EM*np.pi*(2*((-(m0*m1) + m2**2)**2 - (m0**2 + m1**2 + 2*m2**2)*s12 + s12**2) - ((m0 + m1)**2 - 2*s12)*s23 + s23**2) / (256 * np.pi**3 * m0**3) * 1.0 / Br_Gamma_DP_ee(np.sqrt(s23))
    else:
        integrand = lambda s23, s12: 1/Lambdaa**2 * (-16)*ALPHA_EM*np.pi*(2*((-(m0*m1) + m2**2)**2 - (m0**2 + m1**2 + 2*m2**2)*s12 + s12**2) - ((m0 + m1)**2 - 2*s12)*s23 + s23**2) / (256 * np.pi**3 * m0**3)  
    # return custom_dblquad(integrand, s12_min_, s12_max_,
    return custom_dblquad1(integrand, s12_min_, s12_max_,
                          lambda s12: s23_min_s12(s12, m0, m1, m2, m3),
                          lambda s12: s23_max_s12(s12, m0, m1, m2, m3))[0]

def anap_Gamma_chi1_chi0ll_masslessll(m_chi1, m_chi0, Lambdaa=1.0, m_l=M_ELECTRON):
    m0, m1 = m_chi1, m_chi0
    m2 = m3 = m_l

    if m0 < m1 + m2 + m3:
        return 0.0 

    Delta = (m_chi1 - m_chi0) / m_chi0

    achi = 1 / Lambdaa

    return achi**2 *  ALPHA_EM * Delta**5 * m_chi0**5 / (5 * np.pi**2)

def anap_Gamma_chi1_chi0ll_masslesschi0ll(m_chi1, m_chi0, Lambdaa=1.0, m_l=M_ELECTRON):
    m0, m1 = m_chi1, m_chi0
    m2 = m3 = m_l

    if m0 < m1 + m2 + m3:
        return 0.0 

    Delta = (m_chi1 - m_chi0) / m_chi0

    achi = 1 / Lambdaa

    return achi**2 *  ALPHA_EM * m_chi1**5 / (96 * np.pi**2)


def anap_sigma_chi0Nucleus_chi1Nucleus_analyt(Lambdaa, s, m1, m3, Z=Z_TUNGSTEN, A=A_TUNGSTEN, t_max=-1e-15):
    """ chi0(p1) + Nucleus(p2) -> chi1(p3) + Nucleus(p4) """
    m2 = m4 = Z * M_PROTON + (A - Z) * M_NEUTRON  # Nucleus mass in GeV

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if t_max > 0 or t_max < -1: return 0.0
    if t_max > -1e-15: t_max = -1e-15

    a, d = 111.0 * Z**(-1.0 / 3.0) / M_ELECTRON, 0.164 * A**(-2.0 / 3.0)

    # achi = 1 / Lambdaa
    sigma = ALPHAEM * d * Z**2 / Lambdaa**2
    return 0.0 if sigma<0 else sigma

def anap_sigma_chi0e_chi1e_analyt(Lambdaa, s, m1, m3, ERmin, ERmax, t_min, t_max):
    """ chi0 (p1) + e- (p2) -> chi1 (p3) + e- (p4) """
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    ER_min, ER_max = ERmin, ERmax
    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0

    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    sigma = 2 * ALPHAEM * (ER_max - ER_min) * m2 / Lambdaa**2
    return 0.0 if sigma<0 else sigma


def anap_sigma_chi0e_chi1e_log(Lambdaa, s, m1, m3, ERmin, ERmax, t_min, t_max, isFLARE=False):
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    ER_min, ER_max = ERmin, ERmax

    if isFLARE:
        ERmin, ERmax = 30e-3, 1.0
    else:
        ERmin, ERmax = 0.3, 20.0

    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0
    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    log10ER_min, log10ER_max, nlogER = np.log10(ER_min), np.log10(ER_max), 250
    dlog10ER = (log10ER_max - log10ER_min) / float(nlogER)
    sigma = 0

    mchi0 = m1
    mchi1 = m3
    achi = 1 / Lambdaa

    # df  = df / dx * dx = df/dx * dlog10x * x * log10
    for log10ER in np.linspace(log10ER_min, log10ER_max, nlogER):
        ER = 10**log10ER  # ER=E4 is energy recoil of target

        t = 2 * m2 * (m2 - ER)
        u = m1**2 + m2**2 + m3**2 + m4**2 - s - t

        E3 = (m2**2 + m3**2 - u) / (2 * m2)
        p3 = np.sqrt(E3**2 - m3**2)
        pR = np.sqrt(ER**2 - m4**2)

        angle = np.arccos((E3 * ER - 0.5 * (s - m3**2 - m4**2)) / p3 / pR)

        if ER > 10:
            cuts = (10 * 1e-3 < angle < 20 * 1e-3)
        elif 3 < ER < 10:
            cuts = (10 * 1e-3 < angle < 30 * 1e-3)
        elif ER < 3:
            cuts = (10 * 1e-3 < angle)

        if isFLARE: cuts = True

        if not cuts: continue

        amp2 = 8*achi**2*ALPHAEM*Pi*(2*((-(mchi0*mchi1) + M_ELECTRON**2)**2 - (mchi0**2 + mchi1**2 + 2*M_ELECTRON**2)*s + s**2) - ((mchi0 + mchi1)**2 - 2*s)*t + t**2)
                
        dsigmadt = amp2 / (16. * np.pi * (m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s)))

        sigma += 2 * m2 * ER * dsigmadt

    sigma *= dlog10ER * np.log(10)
    return 0.0 if sigma<0 else sigma

def anap_sigma_chi1e_chi0e_log(Lambdaa, s, m1, m3, ERmin, ERmax, t_min, t_max, isFLARE=False):
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    ER_min, ER_max = ERmin, ERmax

    if isFLARE:
        ERmin, ERmax = 30e-3, 1.0
    else:
        ERmin, ERmax = 0.3, 20.0

    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0
    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    log10ER_min, log10ER_max, nlogER = np.log10(ER_min), np.log10(ER_max), 250
    dlog10ER = (log10ER_max - log10ER_min) / float(nlogER)
    sigma = 0

    mchi0 = m1
    mchi1 = m3
    achi = 1 / Lambdaa

    # df  = df / dx * dx = df/dx * dlog10x * x * log10
    for log10ER in np.linspace(log10ER_min, log10ER_max, nlogER):
        ER = 10**log10ER  # ER=E4 is energy recoil of target

        t = 2 * m2 * (m2 - ER)
        u = m1**2 + m2**2 + m3**2 + m4**2 - s - t

        E3 = (m2**2 + m3**2 - u) / (2 * m2)
        p3 = np.sqrt(E3**2 - m3**2)
        pR = np.sqrt(ER**2 - m4**2)

        angle = np.arccos((E3 * ER - 0.5 * (s - m3**2 - m4**2)) / p3 / pR)

        if ER > 10:
            cuts = (10 * 1e-3 < angle < 20 * 1e-3)
        elif 3 < ER < 10:
            cuts = (10 * 1e-3 < angle < 30 * 1e-3)
        elif ER < 3:
            cuts = (10 * 1e-3 < angle)

        if isFLARE: cuts = True

        if not cuts: continue

        amp2 = 8*achi**2*ALPHAEM*Pi*(2*((-(mchi0*mchi1) + M_ELECTRON**2)**2 - (mchi0**2 + mchi1**2 + 2*M_ELECTRON**2)*s + s**2) - ((mchi0 + mchi1)**2 - 2*s)*t + t**2)
                
        dsigmadt = amp2 / (16. * np.pi * (m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s)))

        sigma += 2 * m2 * ER * dsigmadt

    sigma *= dlog10ER * np.log(10)
    return 0.0 if sigma<0 else sigma



## charge-radius DM
def cr_Gamma_chi1_chi0ll(m_chi1, m_chi0, LambdaCR=1.0, m_l=M_ELECTRON, is_Br_Gamma_DP_ee=False):
    m0, m1 = m_chi1, m_chi0
    m2 = m3 = m_l

    if m0 < m1 + m2 + m3:
        return 0.0 

    s12_min_, s12_max_ = (m1 + m2)**2, (m0 - m3)**2

    if is_Br_Gamma_DP_ee:
        integrand = lambda s23, s12: 1/LambdaCR**2 * (-16)*ALPHA_EM * np.pi*(2*(m2**2 - s12)**2 + m0**2*(2*m1**2 - 2*s12 - s23) + 2*s12*s23 + s23**2 + 2*m0*m1*(2*m2**2 + s23) - m1**2*(2*s12 + s23)) / (256 * np.pi**3 * m0**3) * 1.0 / Br_Gamma_DP_ee(np.sqrt(s23))
    else:
        integrand = lambda s23, s12: 1/LambdaCR**2 * (-16)*ALPHA_EM * np.pi*(2*(m2**2 - s12)**2 + m0**2*(2*m1**2 - 2*s12 - s23) + 2*s12*s23 + s23**2 + 2*m0*m1*(2*m2**2 + s23) - m1**2*(2*s12 + s23)) / (256 * np.pi**3 * m0**3)
    # return custom_dblquad(integrand, s12_min_, s12_max_,
    return custom_dblquad1(integrand, s12_min_, s12_max_,
                          lambda s12: s23_min_s12(s12, m0, m1, m2, m3),
                          lambda s12: s23_max_s12(s12, m0, m1, m2, m3))[0]


def cr_Gamma_chi1_chi0ll_masslessll(m_chi1, m_chi0, LambdaCR=1.0, m_l=M_ELECTRON):
    m0, m1 = m_chi1, m_chi0
    m2 = m3 = m_l

    if m0 < m1 + m2 + m3:
        return 0.0 

    Delta = (m_chi1 - m_chi0) / m_chi0

    bchi = 1 / LambdaCR

    return bchi**2 *  ALPHA_EM * Delta**5 * m_chi0**5 / (15 * np.pi**2)

def cr_Gamma_chi1_chi0ll_masslesschi0ll(m_chi1, m_chi0, LambdaCR=1.0, m_l=M_ELECTRON):
    m0, m1 = m_chi1, m_chi0
    m2 = m3 = m_l

    if m0 < m1 + m2 + m3:
        return 0.0 

    Delta = (m_chi1 - m_chi0) / m_chi0

    bchi = 1 / LambdaCR

    return bchi**2 *  ALPHA_EM * m_chi1**5 / (96 * np.pi**2)


def cr_sigma_chi0Nucleus_chi1Nucleus_analyt(LambdaCR, s, m1, m3, Z=Z_TUNGSTEN, A=A_TUNGSTEN, t_max=-1e-15):
    """ chi0(p1) + Nucleus(p2) -> chi1(p3) + Nucleus(p4) """
    m2 = m4 = Z * M_PROTON + (A - Z) * M_NEUTRON  # Nucleus mass in GeV

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if t_max > 0 or t_max < -1: return 0.0
    if t_max > -1e-15: t_max = -1e-15

    a, d = 111.0 * Z**(-1.0 / 3.0) / M_ELECTRON, 0.164 * A**(-2.0 / 3.0)

    sigma = ALPHAEM * d * Z**2 / (LambdaCR**2)
    return 0.0 if sigma<0 else sigma


def cr_sigma_chi0e_chi1e_analyt(LambdaCR, s, m1, m3, ERmin, ERmax, t_min, t_max):
    """ chi0 (p1) + e- (p2) -> chi1 (p3) + e- (p4) """
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    ER_min, ER_max = ERmin, ERmax
    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0

    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    sigma = 2 * ALPHAEM * (ER_max - ER_min) * m2 / LambdaCR**2
    return 0.0 if sigma<0 else sigma


def cr_sigma_chi0e_chi1e_log(LambdaCR, s, m1, m3, ERmin, ERmax, t_min, t_max, isFLARE=False):
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if isFLARE:
        ERmin, ERmax = 30e-3, 1.0
    else:
        ERmin, ERmax = 0.3, 20.0

    ER_min, ER_max = ERmin, ERmax

    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0
    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    log10ER_min, log10ER_max, nlogER = np.log10(ER_min), np.log10(ER_max), 250
    dlog10ER = (log10ER_max - log10ER_min) / float(nlogER)
    sigma = 0

    mchi0 = m1
    mchi1 = m3
    bchi = 1 / LambdaCR

    # df  = df / dx * dx = df/dx * dlog10x * x * log10
    for log10ER in np.linspace(log10ER_min, log10ER_max, nlogER):
        ER = 10**log10ER  # ER=E4 is energy recoil of target

        t = 2 * m2 * (m2 - ER)
        u = m1**2 + m2**2 + m3**2 + m4**2 - s - t

        E3 = (m2**2 + m3**2 - u) / (2 * m2)
        p3 = np.sqrt(E3**2 - m3**2)
        pR = np.sqrt(ER**2 - m4**2)

        angle = np.arccos((E3 * ER - 0.5 * (s - m3**2 - m4**2)) / p3 / pR)

        if ER > 10:
            cuts = (10 * 1e-3 < angle < 20 * 1e-3)
        elif 3 < ER < 10:
            cuts = (10 * 1e-3 < angle < 30 * 1e-3)
        elif ER < 3:
            cuts = (10 * 1e-3 < angle)

        if isFLARE: cuts = True

        if not cuts: continue

        amp2 = 8*ALPHAEM*bchi**2*Pi*(2*(M_ELECTRON**2 - s)**2 + mchi0**2*(2*mchi1**2 - 2*s - t) + 2*s*t + t**2 + 2*mchi0*mchi1*(2*M_ELECTRON**2 + t) - mchi1**2*(2*s + t))
                
        dsigmadt = amp2 / (16. * np.pi * (m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s)))

        sigma += 2 * m2 * ER * dsigmadt

    sigma *= dlog10ER * np.log(10)
    return 0.0 if sigma<0 else sigma

def cr_sigma_chi1e_chi0e_log(LambdaCR, s, m1, m3, ERmin, ERmax, t_min, t_max, isFLARE=False):
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if isFLARE:
        ERmin, ERmax = 30e-3, 1.0
    else:
        ERmin, ERmax = 0.3, 20.0

    ER_min, ER_max = ERmin, ERmax

    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0
    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    log10ER_min, log10ER_max, nlogER = np.log10(ER_min), np.log10(ER_max), 250
    dlog10ER = (log10ER_max - log10ER_min) / float(nlogER)
    sigma = 0

    mchi0 = m1
    mchi1 = m3
    bchi = 1 / LambdaCR

    # df  = df / dx * dx = df/dx * dlog10x * x * log10
    for log10ER in np.linspace(log10ER_min, log10ER_max, nlogER):
        ER = 10**log10ER  # ER=E4 is energy recoil of target

        t = 2 * m2 * (m2 - ER)
        u = m1**2 + m2**2 + m3**2 + m4**2 - s - t

        E3 = (m2**2 + m3**2 - u) / (2 * m2)
        p3 = np.sqrt(E3**2 - m3**2)
        pR = np.sqrt(ER**2 - m4**2)

        angle = np.arccos((E3 * ER - 0.5 * (s - m3**2 - m4**2)) / p3 / pR)

        if ER > 10:
            cuts = (10 * 1e-3 < angle < 20 * 1e-3)
        elif 3 < ER < 10:
            cuts = (10 * 1e-3 < angle < 30 * 1e-3)
        elif ER < 3:
            cuts = (10 * 1e-3 < angle)

        if isFLARE: cuts = True

        if not cuts: continue

        amp2 = 8*ALPHAEM*bchi**2*Pi*(2*(M_ELECTRON**2 - s)**2 + mchi0**2*(2*mchi1**2 - 2*s - t) + 2*s*t + t**2 + 2*mchi0*mchi1*(2*M_ELECTRON**2 + t) - mchi1**2*(2*s + t))
                
        dsigmadt = amp2 / (16. * np.pi * (m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s)))

        sigma += 2 * m2 * ER * dsigmadt

    sigma *= dlog10ER * np.log(10)
    return 0.0 if sigma<0 else sigma



## Axino(atilde)-neutralino(chitilde)-photon
def Gamma_chitilde_gatilde(f_a, m_chitilde, m_atilde, C_agg=1.0):
    m1, m2, m3 = m_chitilde, 0.0, m_atilde
    if m_chitilde < m_atilde:
        return 0.0
    return COS_WEINBERG**2 * (ALPHA_EM**2 * C_agg**2 * (m1**2 - m3**2)**3) / (128. * f_a**2 * m1**3 * np.pi**3)

def Gamma_chitilde_llatilde(f_a, m_chitilde, m_atilde, m_l=M_ELECTRON, C_agg=1.0):
    """
    Args:
        f_a (float): 
        m_chitilde (float): 
        m_atilde (float): 
        m_l (float, optional): . 
        C_agg (float, optional): . 

    Returns:
        float: 
    """
    m0 = m_chitilde
    m1 = m2 = m_l
    m3 = m_atilde

    s12_min_, s12_max_ = (m1 + m2)**2, (m0 - m3)**2

    if m0 < m1 + m2 + m3:
        return 0.0

    integrand = lambda s23, s12: COS_WEINBERG**2 * (ALPHA_EM**3 * C_agg**2 * (2 * m1**4 * s12 + m0**4 * (2 * m1**2 + s12) + 2 * m0 * m3 * s12 * (2 * m1**2 + s12) + s12 * (m3**4 + 2 * s23 * (s12 + s23) - m3**2 * (s12 + 2 * s23)) + 2 * m1**2 * (m3**4 - s12 * (s12 + 2 * s23)) - m0**2 * (4 * m1**2 * m3**2 + s12 * (s12 + 2 * s23)))) / (256. * f_a**2 * m0**3 * np.pi**4 * s12**2)

    return custom_dblquad(integrand, s12_min_, s12_max_,
                          lambda s12: s23_min_s12(s12, m0, m1, m2, m3),
                          lambda s12: s23_max_s12(s12, m0, m1, m2, m3))[0]

def Gamma_chitilde_llatilde_massless_atilde(f_a, m_chitilde, m_atilde, m_l=M_ELECTRON, C_agg=1.0):
    m0 = m_chitilde
    m1 = m2 = m_l
    m3 = m_atilde

    if m0 < m1 + m2 + m3:
        return 0.0

    return -COS_WEINBERG**2 * (ALPHA_EM**3 * C_agg**2 * (2 * m0**6 - 9 * m0**4 * m1**2 + 16 * m1**6 + 3 * m0**6 * np.log( (2 * m1) / m0))) / (576. * f_a**2 * m0**3 * np.pi**4)


def sigma_atildeNucleus_chitildeNucleus_analyt(f_a, s, m1, m3, Z=Z_TUNGSTEN, A=A_TUNGSTEN, C_agg=1.0, t_max=-1e-15):
    """ atilde (p1) + Nucleus (p2) -> chitilde (p3) + Nucleus (p4)
    """
    m2 = m4 = Z * M_PROTON + (A - Z) * M_NEUTRON  # Nucleus mass in GeV

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if t_max > 0 or t_max < -1: return 0.0
    if t_max > -1e-15: t_max = -1e-15

    a, d = 111.0 * Z**(-1.0 / 3.0) / M_ELECTRON, 0.164 * A**(-2.0 / 3.0)
    sigma = COS_WEINBERG**2 * C_agg**2 * ALPHA_EM**3 * Z**2 * (np.log(d / (1 / a**2  - t_max)) - 2.) / (16.0 * np.pi**2 * f_a**2)
    return 0.0 if sigma<0 else sigma

def sigma_atildee_chitildee_analyt(f_a, s, m1, m3, ERmin, ERmax, t_min, t_max, C_agg=1.0):
    m2 = m4 = M_ELECTRON

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    ER_min, ER_max = ERmin, ERmax
    
    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0

    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    sigma = C_agg**2 * ALPHA_EM**3 * np.log(ER_max / ER_min) / (16.0 * np.pi**2 * f_a**2)
    return 0.0 if sigma<0 else COS_WEINBERG**2 * sigma



## Gravitino(gtilde)-neutralino(chitilde)-photon
def Gamma_chitilde_ggtilde(m_chitilde, m_gtilde):
    m0, m1, m2 = m_chitilde, m_gtilde, 0.0
    if m0 < m1 + m2:
        return 0.0

    # # limit for m1->0
    # Gamma_chitilde = (KAPPA**2 * m0**5) / (48. * m1**2 * np.pi)

    # general expression
    Gamma_chitilde = COS_WEINBERG**2 * (KAPPA**2 * (m0**2 - m1**2)**3 * (m0**2 + m1**2)) / (48. * m0**3 * m1**2 * np.pi)
    return Gamma_chitilde

def Gamma_chitilde_llgtilde(m_chitilde, m_gtilde, m_l=M_ELECTRON):
    m0 = m_chitilde
    m1 = m2 = m_l
    m3 = m_gtilde

    s12_min_, s12_max_ = (m1 + m2)**2, (m0 - m3)**2

    if m0 < m1 + m2 + m3:
        return 0.0

    integrand = lambda s23, s12: COS_WEINBERG**2 * (ALPHA_EM * KAPPA**2 * (m0**6 * (2 * m1**2 + s12) + 4 * m0 * m3**3 * s12 * (2 * m1**2 + s12) - m0**4 * (2 * m1**2 * (m3**2 + 2 * s12) + s12 * (-m3**2 + 3 * s12 + 2 * s23)) + (m3**2 - s12) * (2 * m1**4 * s12 + 2 * m1**2 * (m3**4 + m3**2 * s12 - 2 * s12 * s23) + s12 * (m3**4 + s12**2 - 2 * m3**2 * s23 + 2 * s12 * s23 + 2 * s23**2)) + m0**2 * (2 * m1**4 * s12 - 2 * m1**2 * (m3**4 - 2 * m3**2 * s12 - s12 * (s12 - 2 * s23)) + s12 * (m3**4 + 3 * s12**2 + 4 * s12 * s23 + 2 * s23**2 - 2 * m3**2 * (s12 + 2 * s23))))) / (96. * m0**3 * m3**2 * np.pi**2 * s12**2)

    return custom_dblquad(integrand, s12_min_, s12_max_,
                          lambda s12: s23_min_s12(s12, m0, m1, m2, m3),
                          lambda s12: s23_max_s12(s12, m0, m1, m2, m3))[0]

def Gamma_chitilde_llgtilde_massless_gtilde(m_chitilde, m_gtilde, m_l=M_ELECTRON):
    m0 = m_chitilde
    m1 = m2 = m_l
    m3 = m_gtilde

    if m0 < m1 + m2 + m3:
        return 0.0

    return -COS_WEINBERG**2 * (ALPHA_EM * KAPPA**2 * (m0 * np.sqrt(m0**2 - 4 * m1**2) * (15 * m0**6 - 74 * m0**4 * m1**2 - 58 * m0**2 * m1**4 + 36 * m1**6) + 16 * m1**4 * (6 * m0**4 - 16 * m0**2 * m1**2 + 9 * m1**4) * np.arctanh(np.sqrt(1 - (4 * m1**2) / m0**2)) + 8 * (m0**8 - 24 * m0**4 * m1**4) * np.log( (2 * m1) / (m0 + np.sqrt(m0**2 - 4 * m1**2))))) / (576. * m0**3 * m3**2 * np.pi**2)


def sigma_gtildeNucleus_chitildeNucleus_analyt(coupling, s, m1, m3, Z=Z_TUNGSTEN, A=A_TUNGSTEN, t_max=-1e-15):
    """ 
    gtilde(p1) + Nucleus(p2) -> chitilde(p3) + Nucleus(p4)
    """
    m2 = m4 = Z * M_PROTON + (A - Z) * M_NEUTRON  # Nucleus mass in GeV

    # m1 is a dummy variable that gets overwritten:
    # m1 = m_gtilde = (1.0/coupling) / np.sqrt(3) / M_PLANCK_REDUCED       
    F = 1.0 / coupling                       # [coupling = 1/F = 1/GeV**2
                                             # F = np.sqrt(3) * m_gtilde * M_PLANCK_REDUCED] 
    m1 = F / np.sqrt(3) / M_PLANCK_REDUCED   # m_gtilde

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if t_max > 0 or t_max < -1: return 0.0
    if t_max > -1e-15: t_max = -1e-15

    a, d = 111.0 * Z**(-1.0 / 3.0) / M_ELECTRON, 0.164 * A**(-2.0 / 3.0)

    low_mass = (ALPHA_EM * KAPPA**2 * Z**2 * d) / (6*m1**2) if m3 < 1.0 else 0

    high_mass = ALPHA_EM * KAPPA**2 * Z**2 * m3**2 * (np.log(d / (1 / a**2  - t_max)) - 2.) / (6 * m1**2)

    sigma = low_mass + high_mass if high_mass>0 else low_mass
    return 0.0 if sigma<0 else COS_WEINBERG**2 * sigma

def sigma_gtildeNucleus_chitildeNucleus_log(coupling, s, m1, m3, Z=Z_TUNGSTEN, A=A_TUNGSTEN, t_max=-1e-15):
    m2 = m4 = Z * M_PROTON + (A - Z) * M_NEUTRON  # Nucleus mass in GeV

    # m1 is a dummy variable that gets overwritten:
    # m1 = m_gtilde = (1.0/coupling) / np.sqrt(3) / M_PLANCK_REDUCED       
    F = 1.0 / coupling                       # [coupling = 1/F = 1/GeV**2
                                             # F = np.sqrt(3) * m_gtilde * M_PLANCK_REDUCED] 
    m1 = F / np.sqrt(3) / M_PLANCK_REDUCED   # m_gtilde

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if t_max > 0 or t_max < -1: return 0.0
    if t_max > -1e-15: t_max = -1e-15

    t_min = -t_max
    t_max = 1.0

    log10t_min, log10t_max, nlogt = np.log10(t_min), np.log10(t_max), 2000
    dlog10t = (log10t_max - log10t_min) / float(nlogt)
    sigma = 0

    # df  = df / dx * dx = df/dx * dlog10x * x * log10
    for log10t in np.linspace(log10t_min, log10t_max, nlogt):
        t = -10**log10t

        ff2 = Z**2 * np.abs(atomic_nuclear_form_factor(np.sqrt(-t), A, Z))**2
        
        amp2 = (-4*ALPHA_EM*KAPPA**2*np.pi*(3*m1**6*(2*m2**2 + t) + 8*m1**3*m3*t*(2*m2**2 + t) + m1**4*(-10*m2**2*m3**2 + (m3**2 - 6*s - 3*t)*t) + (m3**2 - t)*(2*m2**4*t + 2*m2**2*(m3**4 - (m3**2 + 2*s)*t) + t*(m3**4 + 2*s**2 + 2*s*t + t**2 - 2*m3**2*(s + t))) + m1**2*(6*m2**4*t + t*(3*m3**4 + 6*s**2 + 8*s*t + t**2 - 4*m3**2*(2*s + t)) + 2*m2**2*(m3**4 + 2*m3**2*t - 3*t*(2*s + t))))) / (3.*m1**2*t**2)

        dsigmadt = amp2 / (16. * np.pi * (m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s)))

        sigma += t * ff2 * dsigmadt

    sigma = -sigma * dlog10t * np.log(10)
    return 0.0 if sigma<0 else COS_WEINBERG**2 * sigma

def sigma_gtildee_chitildee_analyt(coupling, s, m1, m3, ERmin, ERmax, t_min, t_max):
    m2 = m4 = M_ELECTRON

    # m1 is a dummy variable that gets overwritten: m1 = m_gtilde = (1.0/coupling) / np.sqrt(3) / M_PLANCK_REDUCED       
    F = 1.0/coupling                         # [coupling = 1/F = 1/GeV**2
                                             # F = np.sqrt(3) * m_gtilde * M_PLANCK_REDUCED] 
    m1 = F/np.sqrt(3)/M_PLANCK_REDUCED       # m_gtilde

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    # m1 is a dummy variable that gets overwritten:
    # m1 = m_gtilde = (1.0/coupling) / np.sqrt(3) / M_PLANCK_REDUCED       
    F = 1.0 / coupling                       # [coupling = 1/F = 1/GeV**2
                                             # F = np.sqrt(3) * m_gtilde * M_PLANCK_REDUCED] 
    m1 = F / np.sqrt(3) / M_PLANCK_REDUCED   # m_gtilde

    ER_min, ER_max = ERmin, ERmax
    
    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0

    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    low = (ALPHAEM * (ER_max - ER_min) * KAPPA**2 * m2) / (3. * m1**2)

    high = (ALPHAEM * np.log(ER_max/ER_min) * KAPPA**2 * m3**2) / (6. * m1**2)

    sigma = high + low if high>0 else low
    return 0.0 if sigma<0 else COS_WEINBERG**2 * sigma

def sigma_gtildee_chitildee_log(coupling, s, m1, m3, ERmin, ERmax, t_min, t_max, isFLARE=False):
    m2 = m4 = M_ELECTRON

    # m1 is a dummy variable that gets overwritten: m1 = m_gtilde = (1.0/coupling) / np.sqrt(3) / M_PLANCK_REDUCED       
    F = 1.0/coupling                         # [coupling = 1/F = 1/GeV**2
                                             # F = np.sqrt(3) * m_gtilde * M_PLANCK_REDUCED] 
    m1 = F/np.sqrt(3)/M_PLANCK_REDUCED       # m_gtilde

    if not s >= (m1 + m2)**2 or not s >= (m3 + m4)**2: return 0

    if isFLARE:
        ERmin, ERmax = 30e-3, 1.0
    else:
        ERmin, ERmax = 0.3, 20.0

    ER_min, ER_max = ERmin, ERmax

    if t_min >= t_max: return 0
    ER_min_t = m2 - t_max / (2 * m2)
    ER_max_t = m2 - t_min / (2 * m2)
    if ER_min_t >= ER_max_t: return 0
    ER_min, ER_max = max(ER_min, ER_min_t), min(ER_max, ER_max_t)
    if ER_min >= ER_max: return 0

    log10ER_min, log10ER_max, nlogER = np.log10(ER_min), np.log10(ER_max), 250
    dlog10ER = (log10ER_max - log10ER_min) / float(nlogER)
    sigma = 0

    # df  = df / dx * dx = df/dx * dlog10x * x * log10
    for log10ER in np.linspace(log10ER_min, log10ER_max, nlogER):
        ER = 10**log10ER  # ER=E4 is energy recoil of target

        t = 2 * m2 * (m2 - ER)
        u = m1**2 + m2**2 + m3**2 + m4**2 - s - t

        E3 = (m2**2 + m3**2 - u) / (2 * m2)
        p3 = np.sqrt(E3**2 - m3**2)
        pR = np.sqrt(ER**2 - m4**2)

        angle = np.arccos((E3 * ER - 0.5 * (s - m3**2 - m4**2)) / p3 / pR)

        if ER > 10:
            cuts = (10 * 1e-3 < angle < 20 * 1e-3)
        elif 3 < ER < 10:
            cuts = (10 * 1e-3 < angle < 30 * 1e-3)
        elif ER < 3:
            cuts = (10 * 1e-3 < angle)

        if isFLARE: cuts = True

        if not cuts: continue

        amp2 = (-4*ALPHA_EM*KAPPA**2*np.pi*(3*m1**6*(2*m2**2 + t) + 8*m1**3*m3*t*(2*m2**2 + t) + m1**4*(-10*m2**2*m3**2 + (m3**2 - 6*s - 3*t)*t) + (m3**2 - t)*(2*m2**4*t + 2*m2**2*(m3**4 - (m3**2 + 2*s)*t) + t*(m3**4 + 2*s**2 + 2*s*t + t**2 - 2*m3**2*(s + t))) + m1**2*(6*m2**4*t + t*(3*m3**4 + 6*s**2 + 8*s*t + t**2 - 4*m3**2*(2*s + t)) + 2*m2**2*(m3**4 + 2*m3**2*t - 3*t*(2*s + t))))) / (3.*m1**2*t**2)

        dsigmadt = amp2 / (16. * np.pi * (m1**4 + (m2**2 - s)**2 - 2*m1**2*(m2**2 + s)))

        sigma += 2 * m2 * ER * dsigmadt

    sigma *= dlog10ER * np.log(10)
    return 0.0 if sigma<0 else COS_WEINBERG**2 * sigma




