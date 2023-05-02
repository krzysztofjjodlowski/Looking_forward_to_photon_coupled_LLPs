#!/usr/bin/env python
# coding: utf-8

from numba import njit  # , jit
import numpy as np
# np.seterr(over='raise')

from scipy.integrate import quad  # , simps, nquad, dblquad
from matplotlib import pyplot as plt

import mpmath
mpmath.mp.dps = 40
mpmath.mp.pretty = False

from constants import *



@njit
def simpson_array(y, a, b, n):
    """Approximates the definite integral of f from a to b by the
    composite Simpson's rule, using n subintervals (with n even)"""
    dx = (b - a) / n
    S = np.sum(y[:-1:2] + 4 * y[1::2] + y[2::2])
    return S * dx / 3


@njit
def Booles_rule_function(f, a, b, zstep=1000):
    """
    f - integrand

    a, b - endpoints
    zstep - number of steps
    """
    delz = (b - a) / zstep
    zvar0 = 0
    sum = 0.0
    for _ in range(zstep):
        zvar1 = zvar0 + 0.25 * delz
        zvar2 = zvar0 + 0.5 * delz
        zvar3 = zvar0 + 0.75 * delz
        zvar4 = zvar0 + delz
        sum += (delz / 90.0) * (7.0 * f(zvar0) + 32.0 * f(zvar1) + 12.0 *
                                 f(zvar2) + 32.0 * f(zvar3) + 7.0 * f(zvar4))
        zvar0 = zvar4
    return sum


@njit
def Booles_rule_array(y, a, b, n=801):
    """
    y - table with integrand evaluated on grid
    x = np.linspace(a,b,n+1)

    a, b - endpoints
    n - number of intervals (so there are n+1 points in total)

    Check https://www.scipress.com/BSMaSS.2.1.pdf


    # @njit
    # def f(x):
    #     return np.exp(x)

    # N = 1000
    # x = np.linspace(0,1,N+1)
    # y = np.array([f(xx) for xx in x])
    # print(Booles_rule_function(f,0,1,zstep=1000))
    # print(Booles_rule_array(y,0,1,N))
    """
    dx = (b - a) / n

    S = 7 * (y[0] + y[-1])
    S += np.sum(32 * y[1::2])
    S += np.sum(12 * y[2:-1:4])
    S += np.sum(14 * y[4:-3:4])
    return S * 2 * dx / 45


# Custom dblquad
def _infunc(x, func, gfun, hfun, more_args):
    a = gfun(x)
    b = hfun(x)
    myargs = (x, ) + more_args
    return quad(func, a, b, args=myargs)[0]


def custom_dblquad(func,
                   a,
                   b,
                   gfun,
                   hfun,
                   args=(),
                   epsabs=0.0,
                   epsrel=1.0e-6,
                   maxp1=100,
                   limit=400):
    return quad(
        _infunc,
        a,
        b,
        (func, gfun, hfun, args),
        epsabs=epsabs,
        epsrel=epsrel,
        maxp1=maxp1,
        limit=limit,
    )

def custom_dblquad1(func,
                    a,
                    b,
                    gfun,
                    hfun,
                    args=(),
                    epsabs=0.0,
                    # epsrel=1.0e-3,
                    # maxp1=50,
                    # limit=200):
                    epsrel=1.0e-2,
                    maxp1=33,
                    limit=133):
    return quad(
        _infunc,
        a,
        b,
        (func, gfun, hfun, args),
        epsabs=epsabs,
        epsrel=epsrel,
        maxp1=maxp1,
        limit=limit,
    )


@njit
def step_Heavyside(x, y):
    if x > 0:
        return 1
    elif x == 0:
        return y
    else:
        return 0


@njit
def lambd(x, y, z):
    """
    Kallen function.
    """
    x, y, z = np.float64(x), np.float64(y), np.float64(z)
    l = (x - y - z)**2 - 4 * y * z
    if l < 0:
        # print(f"Kallen function={l:2.e} for x,y,z={x:2.e},{y:2.e},{z:2.e}")
        # print("Kallen function (less than 0) =  for x,y,z= ")
        # print(l, x, y, z)
        return 0
    return l


# _1 means not-numba implementation
def lambd_1(x, y, z):
    """
    Kallen function.
    """
    l = (x - y - z)**2 - 4 * y * z
    if l < 0:
        # print("Kallen function (less than 0) =  for x,y,z= ")
        # print(l, x, y, z)
        return 0
    return l


@njit
def s23_min_s13(s13, M, m1, m2, m3):
    return (1.0 / (4 * s13) * ((M**2 - m1**2 - m2**2 + m3**2)**2 -
                               (np.sqrt(lambd(M**2, s13, m2**2)) +
                                np.sqrt(lambd(s13, m1**2, m3**2)))**2))


@njit
def s23_max_s13(s13, M, m1, m2, m3):
    return (1.0 / (4 * s13) * ((M**2 - m1**2 - m2**2 + m3**2)**2 -
                               (np.sqrt(lambd(M**2, s13, m2**2)) -
                                np.sqrt(lambd(s13, m1**2, m3**2)))**2))


@njit
def s23_min_s12(s12, M, m1, m2, m3):
    E2 = (s12 - m1**2 + m2**2) / (2 * np.sqrt(s12))
    E3 = (M**2 - s12 - m3**2) / (2 * np.sqrt(s12))
    p2 = np.sqrt(E2**2 - m2**2)
    p3 = np.sqrt(E3**2 - m3**2)

    return (E2 + E3)**2 - (p2 + p3)**2


@njit
def s23_max_s12(s12, M, m1, m2, m3):
    E2 = (s12 - m1**2 + m2**2) / (2 * np.sqrt(s12))
    E3 = (M**2 - s12 - m3**2) / (2 * np.sqrt(s12))
    p2 = np.sqrt(E2**2 - m2**2)
    p3 = np.sqrt(E3**2 - m3**2)

    return (E2 + E3)**2 - (p2 - p3)**2


@njit
def get_t_min_t_max(s, m1, m2, m3, m4):
    s, m1, m2, m3, m4 = np.float64(s), np.float64(m1), np.float64(
        m2), np.float64(m3), np.float64(m4)

    t_min = (
        (m1**2 + m2**2 + m3**2 + m4**2) / 2 - s / 2.0 - 1.0 / (2.0 * s) *
        (m2**2 - m1**2) * (m4**2 - m3**2) - 1.0 /
        (2.0 * s) * np.sqrt(lambd(s, m1**2, m2**2) * lambd(s, m3**2, m4**2)))
    t_max = (
        (m1**2 + m2**2 + m3**2 + m4**2) / 2 - s / 2.0 - 1.0 / (2.0 * s) *
        (m2**2 - m1**2) * (m4**2 - m3**2) + 1.0 /
        (2.0 * s) * np.sqrt(lambd(s, m1**2, m2**2) * lambd(s, m3**2, m4**2)))
    return t_min, t_max


def get_t_min_t_max_1(s, m1, m2, m3, m4):
    s, m1, m2, m3, m4 = mpmath.mpmathify(s), mpmath.mpmathify(
        m1), mpmath.mpmathify(m2), mpmath.mpmathify(m3), mpmath.mpmathify(m4)

    t_min = ((m1**2 + m2**2 + m3**2 + m4**2) / 2 - s / 2.0 - 1.0 / (2.0 * s) *
             (m2**2 - m1**2) * (m4**2 - m3**2) - 1.0 / (2.0 * s) *
             np.sqrt(lambd_1(s, m1**2, m2**2) * lambd_1(s, m3**2, m4**2)))
    t_max = ((m1**2 + m2**2 + m3**2 + m4**2) / 2 - s / 2.0 - 1.0 / (2.0 * s) *
             (m2**2 - m1**2) * (m4**2 - m3**2) + 1.0 / (2.0 * s) *
             np.sqrt(lambd_1(s, m1**2, m2**2) * lambd_1(s, m3**2, m4**2)))
    return np.float64(t_min), np.float64(t_max)



@njit
def BR(Gamma, Gamma_total):
    return Gamma / Gamma_total


def twobody_decay_m2(p0, m0, m1, m2, phi, costheta):
    from skhep.math import LorentzVector

    # Felix version
    # energy and momentum of p2 in the rest frame of p0
    energy = (m0 * m0 - m1 * m1 + m2 * m2) / (2.0 * m0)
    momentum = np.sqrt(energy * energy - m2 * m2)

    # 4-momentum of p2 in the rest frame of p0
    en = energy
    pz = momentum * costheta
    py = momentum * np.sqrt(1.0 - costheta * costheta) * np.sin(phi)
    px = momentum * np.sqrt(1.0 - costheta * costheta) * np.cos(phi)
    p2 = LorentzVector(px, py, pz, en)

    # boost p2 in lab frame
    return p2.boost(-1.0 * p0.boostvector)


@njit
def twobody_decay_m1(p0, m0, m1, m2, phi, costheta):
    """
    p0 is the fourvector of the decaying particle of mass m0; it is also -boostvector that is used to boost p2 particle to LAB frame from p0 frame

    To obtain m2 fourvector, call  m1<->m2;
    """

    # energy and momentum of p2 in the rest frame of p0
    energy = (m0 * m0 + m1 * m1 - m2 * m2) / (2.0 * m0)
    momentum = np.sqrt(energy * energy - m1 * m1)

    # 4-momentum of p2 in the rest frame of p0
    en = energy
    px = momentum * np.sqrt(1.0 - costheta * costheta) * np.cos(phi)
    py = momentum * np.sqrt(1.0 - costheta * costheta) * np.sin(phi)
    pz = momentum * costheta
    p1 = np.array([px, py, pz, en])

    # boost p2 in lab frame
    boostvector = p0 / p0[-1]
    return boost(p1, boostvector)


@njit
def fourvector(energy, m, phi, costheta):
    """
    construct a fourvector
    """
    momentum = np.sqrt(energy**2 - m**2)

    en = energy
    pz = momentum * costheta
    py = momentum * np.sqrt(1.0 - costheta * costheta) * np.sin(phi)
    px = momentum * np.sqrt(1.0 - costheta * costheta) * np.cos(phi)
    return np.array([px, py, pz, en])


@njit
def boost(p, boostvector):
    """
    Boost fourvector p along boostvector.

    Fourvector in the form: (x,y,z,t).  c=1

    Formula from:
    http://physics.unm.edu/Courses/Finley/p581/Handouts/GenLorentzBoostTransfs.pdf
    and also see:
    https://en.wikipedia.org/wiki/Lorentz_transformation#Proper_transformations
    """
    vx, vy, vz = boostvector[:3]
    if not np.any(np.array([vx, vy, vz])):
        return p
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    SpeedOfLight = 1.0

    gamma = 1.0 / np.sqrt(1 - v**2 / SpeedOfLight**2)

    Boost = np.array([
        [
            1 + (gamma - 1) * vx * vx / v**2,
            (gamma - 1) * vx * vy / v**2,
            (gamma - 1) * vx * vz / v**2,
            gamma * vx / SpeedOfLight,
        ],
        [
            (gamma - 1) * vx * vy / v**2,
            1 + (gamma - 1) * vy * vy / v**2,
            (gamma - 1) * vy * vz / v**2,
            gamma * vy / SpeedOfLight,
        ],
        [
            (gamma - 1) * vx * vz / v**2,
            (gamma - 1) * vy * vz / v**2,
            1 + (gamma - 1) * vz * vz / v**2,
            gamma * vz / SpeedOfLight,
        ],
        [
            gamma * vx / SpeedOfLight,
            gamma * vy / SpeedOfLight,
            gamma * vz / SpeedOfLight,
            gamma,
        ],
    ])
    return Boost @ p



@njit
def atomic_nuclear_form_factor(q, A, Z):
    # return 1
    if q <= 0.0:
        return 0
    # Wood-saxon form factor
    R = 1.2 * A**(1.0 / 3.0) * 1e-15  # in meters
    s = 1.0 * 1e-15  # in meters

    # in geV**-1
    r = np.sqrt(R**2 - 5 * s**2) * 5.06773123645372e15

    a, d = ( 111.0 * Z**(-1.0 / 3.0) / M_ELECTRON, 0.164 * A**(-2.0 / 3.0), )

    # # Helm form factor - see, eg., eq A4 2011.04751
    # y = (3 * np.exp(-((q * s)**2) / 2.0) / (q * r)**3 * (np.sin(q * r) - q * r * np.cos(q * r)) )
    y = 1.0

    y *= ( a**2 * q**2 / (1.0 + a**2 * q**2) )  # atomic form factor - see, eg. A5 2011.04751 and code from 2204.03599
    y *= 1.0 / (1.0 + q**2 / d)

    if np.isnan(y):
        y = 0.0
    return y


@njit
def atomic_form_factor(t, A, Z):
    """ t = q**2

    Args:
        t (_type_): _description_

    Returns:
        _type_: _description_
    """
    a, d = (
        111.0 * Z**(-1.0 / 3.0) / M_ELECTRON,
        0.164 * A**(-2.0 / 3.0),
    )

    if t < 7.39 * M_ELECTRON**2:
        return a * a * t / (1. + a * a * t)
    else:
        return 1. / (1. + t / d)

@njit
def get_atomic_a_and_d(A, Z):
    a, d = 111.0 * Z**(-1.0 / 3.0) / M_ELECTRON, 0.164 * A**(-2.0 / 3.0)
    return a, d


def log_interp1d(xx, yy, kind="linear"):
    """
    Linear interpolation in log-log space, ie. if y = a*x then in logspace
    linear function will be fitted
    """
    from scipy.interpolate import interp1d

    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = interp1d(logx, logy, kind=kind,
                          fill_value="extrapolate")  # type: ignore
    return lambda zz: np.power(10.0, lin_interp(np.log10(zz)))


def prepare_plot(figsize=(8, 6),
                 xmin=None,
                 xmax=None,
                 ymin=None,
                 ymax=None,
                 xlabel=None,
                 ylabel=None,
                 title=None,
                 scale_x='log',
                 scale_y='log'):
    fig, ax = plt.subplots(figsize=figsize)

    if xmin and xmax:
        ax.set_xlim(xmin, xmax)
    if ymin and ymax:
        ax.set_ylim(ymin, ymax)

    ax.set_xscale(scale_x)  # type: ignore
    ax.set_yscale(scale_y)  # type: ignore

    if xlabel and ylabel:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.tick_params(axis="x", direction="in", which='both')
    ax.tick_params(axis="y", direction="in", which='both')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    if title:
        ax.title(title)  # type: ignore
    # ax.legend(frameon=1, framealpha=0.5, loc='lower left')
    # ax.grid(alpha=0.25)
    # ax.subplots_adjust(left=0.09, right=0.98, bottom=0.1, top=0.94)
    return fig, ax



def tests():
    a_tun, d_tun = get_atomic_a_and_d(A=A_TUNGSTEN, Z=Z_TUNGSTEN)
    print("get_atomic_a_and_d(A=A_TUNGSTEN, Z=Z_TUNGSTEN) = (%.2e, %.2e)" % get_atomic_a_and_d(A=A_TUNGSTEN, Z=Z_TUNGSTEN))
    print("d_tun, 1/a_tun**2", d_tun, 1/a_tun**2)
    return


if __name__ == "__main__":
    tests()
