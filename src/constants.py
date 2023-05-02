#!/usr/bin/env python
# coding: utf-8

# Global Constants; Everything is in c=hbar=1 units, mass is in GeV units

# Conversion from fortranform to Python
Pi = 3.14159265358979323846

# Constants of nature
N_AVOGADRO = 6.0221409e23
ALPHAEM = ALPHA_EM = 1.0 / 137.035999
E_EM = 0.30282212096          #np.sqrt(ALPHA_EM * 4 * np.pi)
SIN_WEINBERG = 0.47212286536  # np.sqrt(0.22290)
COS_WEINBERG = 0.88153275605  # np.sqrt(1.-0.22290)

M_PROTON = 938.28 * 1e-3
M_NEUTRON = 939.57 * 1e-3
M_ELECTRON = 0.510998950 * 1e-3
M_MUON = 105.6583755 * 1e-3
M_TAU = 1776.86*1e-3

M_PLANCK = 1.220890e19        # GeV
M_PLANCK_REDUCED = 2.435e18   # GeV
KAPPA = 1. / M_PLANCK_REDUCED # 1/GeV - this is needed for gravitino interactions


# pion0, eta, etaprime
M_PI0 = 134.97e-3
F_PI0 = 92.0e-3              # GeV (92 MeV, there is a typo in Schwartz book when defining fPI0)
BR_PI0_gg = 0.98823
M_ETA = 547.862e-3
BR_ETA_gg = 0.3931
M_ETAprime = 957.78e-3
BR_ETAprime_gg = 0.222



# Iron (Fe) - for LHC TAN, SeaQuest, NuCal
RHO_Fe = 7.874             # g/cm**3
A_Fe = 55.845
Z_Fe = 26
M_NUCLEUS_Fe = Z_Fe * M_PROTON + (A_Fe - Z_Fe) * M_NEUTRON  # GeV
Nuc_int_len_Fe = 16.77     # cm
Rad_len_Fe = 1.757         # cm

# Be - for NOMAD, PS191
RHO_Be = 1.848             # g/cm**3
A_Be = 9.0121831                  
Z_Be = 4
M_NUCLEUS_Be = Z_Be * M_PROTON + (A_Be - Z_Be) * M_NEUTRON  # GeV
Nuc_int_len_Be = 42.10     # cm
Rad_len_Be = 35.28         # cm

# C (graphite) - for DUNE
RHO_C = 2.210             # g/cm**3
A_C = 12.0107
Z_C = 6
M_NUCLEUS_C = Z_C * M_PROTON + (A_C - Z_C) * M_NEUTRON  # GeV
Nuc_int_len_C = 42.10  # cm
Rad_len_C = 19.32      # cm

# PDG Standard Rock (Silicon/Si) - for MATHUSLA
RHO_ROCK = 2.650             # g/cm**3
# A_ROCK = 22                  
# Z_ROCK = 11
A_ROCK = 28                  
Z_ROCK = 14
M_NUCLEUS_ROCK = Z_ROCK * M_PROTON + (A_ROCK - Z_ROCK) * M_NEUTRON  # GeV
Nuc_int_len_ROCK = 38.24  # cm
Rad_len_ROCK = 10.02      # cm

# Liquid Argon (LAr) - for FLaRE
RHO_LAr = 1.396             # g/cm**3
A_LAr = 39.948
Z_LAr = 18
M_NUCLEUS_LAr = Z_LAr * M_PROTON + (A_LAr - Z_LAr) * M_NEUTRON  # GeV
Nuc_int_len_LAr = 85.77 # cm
Rad_len_LAr = 14.00     # cm

# Copper (Cu) - for CHARM, NA62
RHO_Cu = 8.96             # g/cm**3
A_Cu = 63.546
Z_Cu = 29
M_NUCLEUS_Cu = Z_Cu * M_PROTON + (A_Cu - Z_Cu) * M_NEUTRON  # GeV
Nuc_int_len_Cu = 15.32 # cm
Rad_len_Cu = 1.436     # cm

# Molybdenum (Mo) - for SHiP
RHO_Mo = 10.22             # g/cm**3
A_Mo = 95.95
Z_Mo = 42
M_NUCLEUS_Mo = Z_Mo * M_PROTON + (A_Mo - Z_Mo) * M_NEUTRON  # GeV
Nuc_int_len_Mo = 15.25  # cm
Rad_len_Mo = 0.9593     # cm

# Tungsten (W) - for FASER2nu
RHO_TUNGSTEN = 19.30         # g/cm**3
A_TUNGSTEN = 183.84
Z_TUNGSTEN = 74
M_NUCLEUS_TUNGSTEN = (Z_TUNGSTEN * M_PROTON + (A_TUNGSTEN - Z_TUNGSTEN) * M_NEUTRON)  # GeV
Nuc_int_len_TUNGSTEN = 9.946   # cm
Rad_len_TUNGSTEN = 0.3504      # cm