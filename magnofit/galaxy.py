from dataclasses import dataclass

import astropy.table
import numpy as np
from astropy import units as u

from . import constants as const
from .calc import luminosity as lc
from .calc import mass as mc


@dataclass
class Galaxy:
    # Total galaxy mass (with dark matter); in Solar masses, converted to code units
    virial_mass: float = 1e12 / const.UNIT_MSUN

    # A property of NFW halos, = ratio of virial and scale radii
    halo_concentration: float = 10.0

    # Gas fraction in the halo
    halo_gas_fraction: float = 1.0e-3

    #Halo density profile
    halo_profile: mc.Mass = mc.MassNFW()

    # Bulge/all gas fraction
    bulge_gas_fraction: float = 0.05

    #Bulge density profile
    bulge_profile: mc.Mass = mc.MassIsothermal()

    # Amoc of outflow as a fraction of a full "sphere"
    outflow_sphere_angle_ratio: float = 1

    # Eddington ratio - AGN luminosity at the start of each episode in Eddington luminosities
    eddington_ratio: float = 1.0
    #Edd ratio at shutdown - how do we define the end of an AGN episode?
    eddington_ratio_shutdown: float = 0.01 #when do we stop outflow driving?

    # Parameters for exponential or powerlaw luminosity decay functions
    drop_timescale: float = 3.0e5 / const.UNIT_YEAR
    alpha_drop: float = 0.5

    # Fraction of time that AGN is active, determines length of pauses between
    # AGN episodes (see quasar_durations).
    duty_cycle: float = 0.05

    # Duration of one full AGN episode (in years, converted to code units)
    # quasar_activity_duration: float = 5e4
    quasar_activity_duration: float = 5e4 / const.UNIT_YEAR

    # SMBH growth timescale at Eddington rate - Salpeter timescale
    salpeter_timescale: float = 4.5e8 * const.RADIATIVE_EFFICIENCY_ETA / const.UNIT_YEAR

    #type of AGN luminosity function
    fade: lc.Luminosity = lc.LuminosityFadeNone()

    smbh_mass: float = None
    bulge_mass: float = None
    bulge_sigma: float = None

    name: str = None

    @property
    def quasar_repetition_timescale(self):
        # Duration between the beginnings of two AGN episodes, to be used when
        # calculating t_eff for luminosity function.
        return self.quasar_activity_duration / self.duty_cycle

    @property
    def virial_radius(self):
        return (626 * (((self.virial_mass / (10 ** 13)) * const.UNIT_MSUN) ** (1 / 3))) / const.UNIT_KPC

    @property
    def bulge_to_total_mass_fraction(self):
        # Fraction of bulge vs whole galaxy mass
        return self.bulge_mass / self.virial_mass

    @property
    def bulge_scale_radius(self):
        return (const.G * self.bulge_mass * const.UNIT_MASS) / (2 * ((self.bulge_sigma * const.UNIT_VELOCITY) ** 2)) / const.UNIT_LENGTH

    @property
    def quasar_lum_variation_timescale(self):
        # Characteristic AGN luminosity change timescale, t_q
        return self.fade.quasar_luminosity_variation_timescale(self)

    @property
    def halo_mass(self):
        return self.virial_mass * (1.0 - self.bulge_to_total_mass_fraction)

    @property
    def halo_scale_radius(self):
        return self.virial_radius / self.halo_concentration

    @property
    def luminosity_eddington(self):
        return const.LUMINOSITY_EDD * (self.smbh_mass * const.UNIT_MSUN) * const.UNIT_TIME / const.UNIT_ENERGY

    def agn_luminosity(self, time):
        return self.fade.luminosity_coefficient(time % self.quasar_repetition_timescale, self) * self.luminosity_eddington

    def to_table(self):
        table = astropy.table.Table(
            {
                "virial_mass": [self.virial_mass * const.UNIT_MSUN] * u.Msun,
                "virial_radius": [self.virial_radius] * u.kpc,
                "bulge_mass": [self.bulge_mass * const.UNIT_MSUN] * u.Msun,
                "bulge_scale": [self.bulge_scale_radius] * u.kpc,
                "bulge_sigma": [self.bulge_sigma] * u.kilometer / u.second,
                "bulge_gas_fraction": [self.bulge_gas_fraction],
                "smbh_mass": [self.smbh_mass * const.UNIT_MSUN] * u.Msun,
                "quasar_activity_duration": [self.quasar_activity_duration * const.UNIT_YEAR] * u.year,
                "duty_cycle": [self.duty_cycle],
                "fade_type": [str(self.fade)],
                "outflow_solid_angle_fraction": [self.outflow_sphere_angle_ratio],
            },
        )
        if self.name:
            table["name"] = [self.name]

        return table


    def generate_stochastic_parameters(self, rng):
        if self.virial_mass is None:
            self.virial_mass = self._virial_mass()

        if self.smbh_mass is None:
            self.smbh_mass = self._smbh_mass(rng)

        if self.bulge_mass is None:
            self.bulge_mass = self._bulge_mass(rng)

        if self.bulge_sigma is None:
            self.bulge_sigma = self._bulge_sigma(rng)

    def _smbh_mass(self, rng):
        # Calculate SMBH mass from total galaxy mass (including dark matter)
        # Bandara et al. 2009, doi: 10.1088/0004-637X/704/2/1135 (equation 8)
        # We randomize the value of the first free coefficient to provide
        # a realistic spread of smbh masses. Note that the bounds here
        # are somewhat smaller than in the original equation and are
        # sampled uniformly.
        free_coef = 8.18 + rng.uniform(-0.4, 0.4)
        log_smbh_mass = free_coef + (1.55 * (np.log10(self.virial_mass * const.UNIT_MSUN) - 13.0))
        return (10 ** log_smbh_mass) / const.UNIT_MSUN

    def _bulge_mass(self, rng):
        # Calculate bulge mass from SMBH mass, included variance mimics intrinsic scatter
        # McConnell & Ma 2013, doi: 10.1088/0004-637X/764/2/184 (see abstract)
        # We randomize the value of the first free coefficient to provide
        # a realistic spread of bulge masses. Note that the bounds here
        # were derived by visually inspecting the results.
        intercept_alpha = 8.46 + rng.uniform(-0.34, 0.34)
        slope_beta = 1.05
        log_bulge_mass = (np.log10(self.smbh_mass * const.UNIT_MSUN) - intercept_alpha) / slope_beta
        return ((10 ** log_bulge_mass) * 1e11) / const.UNIT_MSUN

    def _bulge_sigma(self, rng):
        # Calculate bulge sigma from SMBH mass
        # McConnell & Ma 2013, doi: 10.1088/0004-637X/764/2/184 (see abstract)
        # We randomize the value of the first free coefficient to provide
        # a realistic spread of bulge masses. Note that the bounds here
        # were derived by visually inspecting the results.
        intercept_alpha = 8.32 + rng.uniform(-0.38, 0.38)
        slope_beta = 5.64
        log_sigma = (np.log10(self.smbh_mass * const.UNIT_MSUN) - intercept_alpha) / slope_beta
        return ((10 ** log_sigma) * 2e7) / const.UNIT_VELOCITY

    def _virial_mass(self):
        # Bandara et al. 2009, doi: 10.1088/0004-637X/704/2/1135 (equation 8)
        log_virial_mass = ((np.log10(self.smbh_mass* const.UNIT_MSUN) - 8.18) / 1.55 ) + 13.0
        return (10 ** log_virial_mass) / const.UNIT_MSUN
