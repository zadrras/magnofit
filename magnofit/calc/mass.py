import numpy as np


class Mass:
    def calculate(self, radius, dot_radius, dotdot_radius, total_mass, scale_length, concentration, gas_fraction):
        #Scale the radius to the size of the component:
        scaled_radius = radius/scale_length
        scaled_dot_radius = dot_radius/scale_length
        scaled_dotdot_radius = dotdot_radius/scale_length

        # Calculate fraction of mass within the scale and the derivatives
        (
            mass_fraction,
            dot_mass_fraction,
            dotdot_mass_fraction,
            rho_contact,
            rho_outer,
        ) = self.calculate_fractions(
            scaled_radius, scaled_dot_radius, scaled_dotdot_radius, concentration
        )

        mass_potential = total_mass * mass_fraction * (1 - gas_fraction)
        dot_mass_potential = total_mass * dot_mass_fraction * (1 - gas_fraction)
        mass_gas = total_mass * mass_fraction * gas_fraction
        dot_mass_gas = total_mass * dot_mass_fraction * gas_fraction
        dotdot_mass_gas = total_mass * dotdot_mass_fraction * gas_fraction
        rho_gas_contact = 3. * total_mass / (4. * np.pi * scale_length ** 3) * rho_contact * gas_fraction
        rho_gas_outer = rho_gas_contact / rho_contact * rho_outer * gas_fraction

        return mass_potential, dot_mass_potential, mass_gas, dot_mass_gas, dotdot_mass_gas, rho_gas_contact, rho_gas_outer

    def calculate_fractions(scaled_radius, scaled_dot_radius, scaled_dotdot_radius, concentration):
        raise NotImplementedError()


class MassNFW(Mass):
    def calculate_fractions(self, scaled_radius, scaled_dot_radius, scaled_dotdot_radius, concentration):
        '''
        Parameters
        ----------
        scaled_radius : float; ratio of outflow radius to scale_length
        scaled_dot_radius : float; ratio of outflow velocity to scale_length
        scaled_dotdot_radius : float; ratio of outflow acceleration to scale_length
        concentration : float; concentration (= R_vir / scale_length) of the NFW profile

        Returns: mass_fraction, dot_mass_fraction, dotdot_mass_fraction, rho_contact, rho_outer

        '''

        concentration_term = np.log(1. + concentration) - concentration/(1. + concentration)

        mass_fraction = (np.log(1. + scaled_radius) - scaled_radius/(1. + scaled_radius)) / concentration_term
        dot_mass_fraction = scaled_dot_radius * scaled_radius / ((1. + scaled_radius) ** 2) / concentration_term
        dotdot_mass_fraction = (scaled_dotdot_radius * scaled_radius / ((1. + scaled_radius) ** 2) + scaled_dot_radius ** 2 * (1. - scaled_radius) / ((1. + scaled_radius) ** 3)) / concentration_term

        rho_contact = ((1. + scaled_radius) ** 2) / (3. * scaled_radius) / concentration_term

        bigger_radius = 4. * scaled_radius / 3.

        rho_outer = ((1. + bigger_radius) ** 2) / (3. * bigger_radius) / concentration_term

        if mass_fraction > 1.:
            mass_fraction = 1.
            dot_mass_fraction = 0.
            dotdot_mass_fraction = 0.
            rho_contact = 1.e-10
            rho_outer = 1.e-10

        return mass_fraction, dot_mass_fraction, dotdot_mass_fraction, rho_contact, rho_outer


class MassIsothermal(Mass):
    def calculate_fractions(self, scaled_radius, scaled_dot_radius, scaled_dotdot_radius, concentration):
        '''

        Parameters
        ----------
        scaled_radius : float; ratio of outflow radius to scale_length
        scaled_dot_radius : float; ratio of outflow velocity to scale_length
        scaled_dotdot_radius : float; ratio of outflow acceleration to scale_length
        concentration : dummy parameter, not required for this profile

        Returns: mass_fraction, dot_mass_fraction, dotdot_mass_fraction, rho_contact, rho_outer

        '''

        mass_fraction = scaled_radius
        dot_mass_fraction = scaled_dot_radius
        dotdot_mass_fraction = scaled_dotdot_radius

        rho_contact = 1./3.
        rho_outer = 1./3. * (3./4.) ** 3

        if mass_fraction > 1.:
            mass_fraction = 1.
            dot_mass_fraction = 0.
            dotdot_mass_fraction = 0.
            rho_contact = 1.e-10
            rho_outer = 1.e-10


        return mass_fraction, dot_mass_fraction, dotdot_mass_fraction, rho_contact, rho_outer


class MassHernquist(Mass):
    def calculate_fractions(self, scaled_radius, scaled_dot_radius, scaled_dotdot_radius, concentration):
        '''

        Parameters
        ----------
        scaled_radius : float; ratio of outflow radius to scale_length
        scaled_dot_radius : float; ratio of outflow velocity to scale_length
        scaled_dotdot_radius : float; ratio of outflow acceleration to scale_length
        concentration : dummy parameter, not required for this profile

        Returns: mass_fraction, dot_mass_fraction, dotdot_mass_fraction, rho_contact, rho_outer

        '''

        mass_fraction = scaled_radius ** 2. / ((1. + scaled_radius) ** 2.)
        dot_mass_fraction = scaled_dot_radius * 2. * scaled_radius / ((1. + scaled_radius) ** 3.)
        dotdot_mass_fraction = 2. * (scaled_dotdot_radius * scaled_radius + scaled_dot_radius ** 2. * (1. - 2. * scaled_radius) / (1. + scaled_radius)) / ((1. + scaled_radius) ** 3.)

        rho_contact = 2./3. / scaled_radius / ((1. + scaled_radius) ** 3.)
        bigger_radius = 4. * scaled_radius / 3.
        rho_outer = 2./3. / bigger_radius / ((1. + bigger_radius) ** 3.)

        return mass_fraction, dot_mass_fraction, dotdot_mass_fraction, rho_contact, rho_outer


class MassJaffe(Mass):
    def calculate_fractions(self, scaled_radius, scaled_dot_radius, scaled_dotdot_radius, concentration):
        '''

        Parameters
        ----------
        scaled_radius : float; ratio of outflow radius to scale_length
        scaled_dot_radius : float; ratio of outflow velocity to scale_length
        scaled_dotdot_radius : float; ratio of outflow acceleration to scale_length
        concentration : dummy parameter, not required for this profile

        Returns: mass_fraction, dot_mass_fraction, dotdot_mass_fraction, rho_contact, rho_outer

        '''

        mass_fraction = scaled_radius / (1. + scaled_radius)
        dot_mass_fraction = scaled_dot_radius / ((1. + scaled_radius) ** 2.)
        dotdot_mass_fraction = scaled_dotdot_radius / ((1. + scaled_radius) ** 2.) + scaled_dot_radius ** 2. * 2. / ((1. + scaled_radius) ** 3.)

        rho_contact = 1./3. / (scaled_radius ** 2.) / ((1. + scaled_radius) ** 2.)
        bigger_radius = 4. * scaled_radius / 3.
        rho_outer = 1./3. / (bigger_radius ** 2.) / ((1. + bigger_radius) ** 2.)

        return mass_fraction, dot_mass_fraction, dotdot_mass_fraction, rho_contact, rho_outer
