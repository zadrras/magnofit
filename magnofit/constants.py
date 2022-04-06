import numpy as np


_kpc = 3.086e21
_sunmass = 1.989e33  # [g]
_opacity = 0.346  # electron scattering opacity for Solar metallicity [g / cm^-2]

G = 6.672e-8  # Gravitational constant
c = 2.9979e10  # Light speed [cm/s]

SECONDS_IN_YEAR = 31556952

LUMINOSITY_EDD = 4 * np.pi * G * _sunmass * c / _opacity

# coupling efficiency between luminosity or momentum and driving power/force
ETA_DRIVE = 0.05
RADIATIVE_EFFICIENCY_ETA = 0.1  # radiative efficiency
GAMMA = 5.0 / 3.0  # adiabatic index of the outflowing material

# All of the following measures are for conversion between simulation and real units
UNIT_LENGTH = 3.086e21
UNIT_MASS = 5e9 * _sunmass
UNIT_VELOCITY = np.sqrt(G * UNIT_MASS / UNIT_LENGTH)

UNIT_TIME = UNIT_LENGTH / UNIT_VELOCITY
UNIT_ENERGY = UNIT_MASS * (UNIT_VELOCITY ** 2)

UNIT_KPC = UNIT_LENGTH / _kpc
UNIT_YEAR = UNIT_TIME / SECONDS_IN_YEAR
UNIT_MSUN = UNIT_MASS / _sunmass