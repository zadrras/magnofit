import numpy as np


class Luminosity:
    def luminosity_coefficient(self, time_eff, galaxy):
        pass

    def luminosity_mean_coefficient(self, time_start, duration, timestep, galaxy):
        pass

    def quasar_luminosity_variation_timescale(self, galaxy):
        pass

    def __str__(self):
        return self.__class__.__name__


class LuminosityFadeNone(Luminosity):
    def luminosity_coefficient(self, time_eff, galaxy):
        coef = 0.0
        if time_eff <= galaxy.quasar_lum_variation_timescale:
            coef = galaxy.eddington_ratio

        return coef

    def luminosity_mean_coefficient(self, time_start, duration, timestep, galaxy):
        return galaxy.eddington_ratio * duration / timestep

    def quasar_luminosity_variation_timescale(self, galaxy):
        return galaxy.quasar_activity_duration


class LuminosityFadeExponential(Luminosity):
    def luminosity_coefficient(self, time_eff, galaxy):
        if time_eff <= galaxy.quasar_lum_variation_timescale:
            coef = galaxy.eddington_ratio
        else:
            coef = galaxy.eddington_ratio * np.exp(-(time_eff - galaxy.quasar_lum_variation_timescale) / galaxy.drop_timescale)

        return coef

    def luminosity_mean_coefficient(self, time_start, duration, timestep, galaxy):
        if time_start + duration <= galaxy.quasar_lum_variation_timescale:
            coef = galaxy.eddington_ratio
        else:
            if time_start <= galaxy.quasar_lum_variation_timescale:
                coef = galaxy.eddington_ratio * (galaxy.quasar_lum_variation_timescale - time_start) / timestep + galaxy.eddington_ratio * galaxy.drop_timescale / timestep * (1 - np.exp ((galaxy.quasar_lum_variation_timescale - duration - time_start) / galaxy.drop_timescale))
            else:
                coef = galaxy.eddington_ratio * galaxy.drop_timescale / timestep * np.exp((galaxy.quasar_lum_variation_timescale - time_start) / galaxy.drop_timescale) * (1 - np.exp (- duration / galaxy.drop_timescale))

        return coef

    def quasar_luminosity_variation_timescale(self, galaxy):
        return galaxy.quasar_activity_duration + galaxy.drop_timescale * np.log(galaxy.eddington_ratio_shutdown/galaxy.eddington_ratio)


class LuminosityFadePowerLaw(Luminosity):
    def luminosity_coefficient(self, time_eff, galaxy):
        if time_eff <= galaxy.quasar_lum_variation_timescale:
            coef = galaxy.eddington_ratio
        else:
            coef =  galaxy.eddington_ratio * (time_eff / galaxy.quasar_lum_variation_timescale) ** (-1. * galaxy.alpha_drop)

        return coef

    def luminosity_mean_coefficient(self, time_start, duration, timestep, galaxy):
        if time_start + duration <= galaxy.quasar_lum_variation_timescale:
            coef = galaxy.eddington_ratio
        else:
            if galaxy.alpha_drop == -1: #special case, integral becomes logarithmic
                if time_start <= galaxy.quasar_lum_variation_timescale:
                    coef = galaxy.eddington_ratio * (galaxy.quasar_lum_variation_timescale - time_start) / timestep + galaxy.eddington_ratio * galaxy.quasar_lum_variation_timescale / timestep * np.log((time_start + duration) / galaxy.quasar_lum_variation_timescale)
                else:
                    galaxy.eddington_ratio * galaxy.quasar_lum_variation_timescale / timestep * np.log((time_start + duration) / time_start)
            else:
                if time_start <= galaxy.quasar_lum_variation_timescale:
                    coef = galaxy.eddington_ratio * (galaxy.quasar_lum_variation_timescale - time_start) / timestep + galaxy.eddington_ratio * galaxy.quasar_lum_variation_timescale ** galaxy.alpha_drop / (timestep * (1. - galaxy.alpha_drop)) * (galaxy.quasar_lum_variation_timescale ** (1. - galaxy.alpha_drop) - (time_start + duration) ** (1. - galaxy.alpha_drop))
                else:
                    coef = galaxy.eddington_ratio * galaxy.quasar_lum_variation_timescale ** galaxy.alpha_drop / (timestep * (1. - galaxy.alpha_drop)) * (time_start ** (1. - galaxy.alpha_drop) - (time_start + duration) ** (1. - galaxy.alpha_drop))

        return coef

    def quasar_luminosity_variation_timescale(self, galaxy):
        return galaxy.quasar_activity_duration / (1 + (galaxy.eddington_ratio_shutdown/galaxy.eddington_ratio) ** (-1. / galaxy.alpha_drop))


class LuminosityFadeKing(Luminosity):
    def luminosity_coefficient(self, time_eff, galaxy):
        coef = galaxy.eddington_ratio * (1 + time_eff / galaxy.quasar_lum_variation_timescale) ** (-19. / 16.)
        return coef

    def luminosity_mean_coefficient(self, time_start, duration, timestep, galaxy):
        return galaxy.eddington_ratio * 16 * galaxy.quasar_lum_variation_timescale / (3 * timestep) * ((1 + time_start / galaxy.quasar_lum_variation_timescale) ** (-3./16.) - (1 + (time_start + duration) / galaxy.quasar_lum_variation_timescale) ** (-3./16.))


    def quasar_luminosity_variation_timescale(self, galaxy):
        return galaxy.quasar_activity_duration / ((galaxy.eddington_ratio_shutdown/galaxy.eddington_ratio) ** (-16. / 19.) - 1.)
