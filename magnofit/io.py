import astropy.table
import numpy as np
from astropy import units as u

import magnofit.constants as const


def outflows_to_table(outflows, galaxy_params):
    outflow_array = np.empty(
        (len(outflows),),
        dtype=np.dtype(
            [
                ("time", np.float64),
                ("dot_time", np.float64),
                ("radius", np.float64),
                ("dot_radius", np.float64),
                ("dotdot_radius", np.float64),
                ("dotdotdot_radius", np.float64),
                ("dot_mass", np.float64),
                ("mass_out", np.float64),
                ("total_mass", np.float64),
                ("luminosity_AGN", np.float64),
            ]
        ),
    )

    for i, (o, g) in enumerate(zip(outflows, galaxy_params)):
        outflow_array["radius"][i] = o.radius
        outflow_array["dot_radius"][i] = o.dot_radius
        outflow_array["dotdot_radius"][i] = o.dotdot_radius
        outflow_array["dotdotdot_radius"][i] = o.dotdotdot_radius

        outflow_array["time"][i] = o.time
        outflow_array["dot_time"][i] = o.dot_time

        outflow_array["mass_out"][i] = o.mass_out
        outflow_array["total_mass"][i] = o.total_mass

        outflow_array["luminosity_AGN"][i] = g.agn_luminosity(o.time)

    outflow_array["radius"] = outflow_array["radius"] * const.UNIT_KPC
    outflow_array["dot_radius"] = (
        outflow_array["dot_radius"] * const.UNIT_VELOCITY / 1.0e5
    )  # to km/s
    outflow_array["time"] = outflow_array["time"] * const.UNIT_YEAR
    outflow_array["dot_time"] = outflow_array["dot_time"] * const.UNIT_YEAR

    outflow_array["mass_out"] = (
        outflow_array["mass_out"]
        * galaxy_params[0].outflow_sphere_angle_ratio
        * const.UNIT_MSUN
    )
    derived_dot_mass = np.divide(
        outflow_array["mass_out"] * outflow_array["dot_radius"] * 1.0e5,  # to cm/s
        outflow_array["radius"] * const.UNIT_LENGTH,
        out=np.zeros_like(outflow_array["radius"]),
        where=(outflow_array["radius"] != 0.0),
    )
    outflow_array["dot_mass"] = derived_dot_mass * const.SECONDS_IN_YEAR

    outflow_array["total_mass"] = outflow_array["total_mass"] * const.UNIT_MSUN
    outflow_array["luminosity_AGN"] = (
        outflow_array["luminosity_AGN"] * const.UNIT_ENERGY / const.UNIT_TIME
    )

    outflow_table = astropy.table.Table(
        outflow_array,
        units=(
            u.yr,
            u.yr,
            u.kpc,
            u.km / u.s,
            u.km / u.s / u.s,
            u.km / u.s / u.s / u.s,
            u.Msun / u.yr,
            u.Msun,
            u.Msun,
            u.erg / u.s,
        ),
        descriptions=(
            "Time since the start of the outflow",
            "Current timestep",
            "Current outflow radius",
            "Current outflow velocity",
            "Current outflow acceleration",
            "Current outflow jerk",
            "Mass outflow rate derived as M v / R",
            "Gas mass in outflow",
            "Total mass within outflow radius",
            "Current AGN luminosity",
        ),
    )

    return outflow_table
