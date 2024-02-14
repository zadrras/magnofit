import dataclasses
from copy import copy

import numpy as np

from . import constants as const
from . import io
from .calc import time as tc
from .galaxy import Galaxy


@dataclasses.dataclass
class OutflowState:
    radius: float = 0.0
    dot_radius: float = 0.0
    dotdot_radius: float = 0.0
    dotdotdot_radius: float = 0.0

    # Total mass of outflowing gas
    mass_out: float = 0.0
    # Total mass within outflow radius, including gas and non-gas ('potential') components
    total_mass: float = 0.0
    # Mass outflow rate per unit time
    dot_mass: float = 0.0

    time: float = 0.0
    dot_time: float = 0.0


def run_outflow_simulation(
    init_params: Galaxy,
    output_array_length=200,
    smbh_grows=True,
    max_timesteps=30000,
    max_time=1.5e8 / const.UNIT_YEAR,
    max_radius=12.0 / const.UNIT_KPC,
    dt_min=1.0 / const.UNIT_YEAR,
    rng=np.random.default_rng(0),
):
    dtmax = init_params.quasar_activity_duration * 0.1
    agn_episode_start_flag = 0
    courant_factor = 0.02

    curr_outflow = OutflowState(
        radius=0.001 / const.UNIT_KPC,
        dot_radius=100000.0 / const.UNIT_VELOCITY,
    )
    curr_galaxy = copy(init_params)

    outflows = []
    galaxy_params = []

    timestep = 0
    while (
        timestep < max_timesteps - 1
        and curr_outflow.time < max_time
        and curr_outflow.radius < max_radius
    ):
        timestep += 1
        """Main simulation loop. Essentially, this consists of three steps:
        1. Calculate the potential and gas masses within the current outflow radius
        2. Determine the appropriate timestep
        3. Propagate the outflow to the new radius, recording all the relevant quantities
        """

        # Mass calculation adds up two components: bulge and halo.
        # More components can be added in a straightforward manner.
        halo_component = init_params.halo_profile.calculate(
            curr_outflow.radius,
            curr_outflow.dot_radius,
            curr_outflow.dotdot_radius,
            init_params.halo_mass,
            init_params.halo_scale_radius,
            init_params.halo_concentration,
            init_params.halo_gas_fraction,
        )
        bulge_component = init_params.bulge_profile.calculate(
            curr_outflow.radius,
            curr_outflow.dot_radius,
            curr_outflow.dotdot_radius,
            init_params.bulge_mass,
            init_params.bulge_scale_radius,
            None,
            init_params.bulge_gas_fraction,
        )

        (
            mass_potential,
            dot_mass_potential,
            mass_gas,
            dot_mass_gas,
            dotdot_mass_gas,
            _,
            _,
        ) = (a + b for a, b in zip(halo_component, bulge_component))

        # Total mass of outflowing gas
        curr_outflow.mass_out = mass_gas
        # Total mass within outflow radius, including gas and non-gas ('potential') components
        curr_outflow.total_mass = mass_gas + mass_potential
        # Mass outflow rate per unit time
        curr_outflow.dot_mass = dot_mass_gas

        # Calculation of timestep, using a Courant-like criterion
        # radius / velocity
        dot_t1 = curr_outflow.radius / (
            abs(curr_outflow.dot_radius) + np.finfo(float).eps
        )
        # velocity / acceleration
        dot_t2 = curr_outflow.dot_radius / (
            abs(curr_outflow.dotdot_radius) + np.finfo(float).eps
        )
        # acceleration / jerk
        dot_t3 = curr_outflow.dotdot_radius / (
            abs(curr_outflow.dotdotdot_radius) + np.finfo(float).eps
        )

        # Most conservative time step size
        dt = courant_factor * min(abs(dot_t1), abs(dot_t2), abs(dot_t3))

        # We have to be careful at the start of each AGN episode in order to
        # propagate the derivatives of radius properly.
        if agn_episode_start_flag > 0:
            dt = dt_min
            agn_episode_start_flag += 1
            if agn_episode_start_flag > 3:
                agn_episode_start_flag = 0

        # If we're jumping to a new AGN episode, shorten the timestep to coincide
        # with exactly the start of the episode, in order to get correct luminosity output
        next_rep = (curr_outflow.time + dt) // init_params.quasar_repetition_timescale
        curr_rep = curr_outflow.time // init_params.quasar_repetition_timescale
        if next_rep > curr_rep:
            agn_episode_start_flag = 1
            dt = (
                init_params.quasar_repetition_timescale
                * ((curr_outflow.time + dt) // init_params.quasar_repetition_timescale)
                - curr_outflow.time
                + np.finfo(float).eps
            )  # want an epsilon to make sure we are inside an AGN episode now

        dt = max(dt, dt_min)
        dt = min(dt, dtmax)

        curr_outflow.dot_time = dt
        next_outflow = OutflowState(time=curr_outflow.time + dt)

        time_eff = curr_outflow.time % init_params.quasar_repetition_timescale

        # Calculation of "driving" luminosity
        mean_luminosity_coef = init_params.fade.luminosity_coefficient(
            time_eff, init_params
        )

        # We are either fully outside an AGN episode or just before one's start;
        # in the latter case, we will spend ~eps time in the episode, no energy injection will occur
        if mean_luminosity_coef >= init_params.eddington_ratio_shutdown:
            # This is the expected luminosity coefficient at the end of this timestep
            predicted_luminosity_coef = init_params.fade.luminosity_coefficient(
                time_eff + dt, init_params
            )
            # We are fully inside an AGN episode
            if predicted_luminosity_coef >= init_params.eddington_ratio_shutdown:
                mean_luminosity_coef = init_params.fade.luminosity_mean_coefficient(
                    # effective time now
                    time_eff,
                    # duration of activity during this timestep
                    dt,
                    # length of this timestep
                    dt,
                    init_params,
                )
            # We are passing the end of an AGN episode
            else:
                mean_luminosity_coef = init_params.fade.luminosity_mean_coefficient(
                    # effective time now
                    time_eff,
                    # duration for which the AGN is still active during this timestep
                    init_params.quasar_activity_duration - time_eff,
                    # length of this timestep
                    dt,
                    init_params,
                )
        else:
            mean_luminosity_coef = 0.0

        mean_luminosity = mean_luminosity_coef * curr_galaxy.luminosity_eddington

        next_galaxy = copy(curr_galaxy)
        if smbh_grows:
            next_galaxy.smbh_mass *= np.exp(
                mean_luminosity_coef * dt / init_params.salpeter_timescale
            )

        # Calculates next radius and its derivatives from various current parameters
        (
            next_outflow.radius,
            next_outflow.dot_radius,
            next_outflow.dotdot_radius,
            next_outflow.dotdotdot_radius,
        ) = tc.simple_time_step(
            curr_outflow.radius,
            curr_outflow.dot_radius,
            curr_outflow.dotdot_radius,
            curr_outflow.dotdotdot_radius,
            mass_potential,
            dot_mass_potential,
            mass_gas,
            dot_mass_gas,
            dotdot_mass_gas,
            mean_luminosity,
            dt,
        )

        outflows.append(curr_outflow)
        galaxy_params.append(curr_galaxy)

        if next_outflow.radius < 0.0:
            print(
                f"At step = {timestep} time = {curr_outflow.time} calc failed due to negative radius."
            )
            return None

        curr_outflow = next_outflow
        next_outflow = OutflowState()

        curr_galaxy = next_galaxy
        next_galaxy = copy(curr_galaxy)

    # Reject outflows with radius <= 0.02
    galaxy_params = [g for (o, g) in zip(outflows, galaxy_params) if o.radius > 0.02]
    outflows = [o for o in outflows if o.radius > 0.02]

    # Randomly select predefined number of outflows
    # The pairing between outflows and galaxies must be retained, hence the syntax
    if rng and len(outflows) > 0:
        weights = [o.dot_time for o in outflows]
        weights /= np.sum(weights)
        size = min(len(outflows), output_array_length)
        outflows, galaxy_params = zip(
            *rng.choice(list(zip(outflows, galaxy_params)), p=weights, size=size)
        )
    else:
        return None

    return io.outflows_to_table(outflows, galaxy_params)
