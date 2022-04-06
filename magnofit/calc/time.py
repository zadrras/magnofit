from .. import constants as const


def simple_time_step(radius, dot_radius, dotdot_radius, dotdotdot_radius, mass_potential, dot_mass_potential, mass_gas, dot_mass_gas, dotdot_mass_gas, luminosity, dt):
    dotdotdot_radius_new = rtdot_calc(luminosity, mass_gas, dot_mass_gas, dotdot_mass_gas, mass_potential, dot_mass_potential, radius, dot_radius, dotdot_radius)
    dotdot_radius_new = dotdot_radius + dotdotdot_radius_new * dt
    dot_radius_new = dot_radius + dotdot_radius_new * dt + 0.5 * dotdotdot_radius_new * (dt ** 2.)
    dot_radius_new, dotdot_radius_new, dotdotdot_radius_new = cap_velocities(dot_radius_new, dotdot_radius_new, dotdotdot_radius_new)
    radius_new = radius + dot_radius_new * dt + 0.5 * dotdot_radius_new * dt ** 2. + (1. / 6.) * dotdotdot_radius_new * dt ** 3.

    return radius_new, dot_radius_new, dotdot_radius_new, dotdotdot_radius_new


def leapfrog_dkd_time_step(radius, dot_radius, dotdot_radius, dotdotdot_radius, mass_potential, dot_mass_potential, mass_gas, dot_mass_gas, dotdot_mass_gas, luminosity, dt):
    #drift half step
    intermediate_dotdot_radius = dotdot_radius + dotdotdot_radius * dt / 2
    intermediate_radius = radius + dot_radius * dt / 2

    #kick full step
    dotdotdot_radius_new = rtdot_calc(luminosity, mass_gas, dot_mass_gas, dotdot_mass_gas, mass_potential, dot_mass_potential, intermediate_radius, dot_radius, intermediate_dotdot_radius)
    dot_radius_new = dot_radius + intermediate_dotdot_radius * dt
    dot_radius_new, intermediate_dotdot_radius, dotdotdot_radius_new = cap_velocities(dot_radius_new, intermediate_dotdot_radius, dotdotdot_radius_new)

    #drift half step
    dotdot_radius_new = intermediate_dotdot_radius + dotdotdot_radius_new * dt / 2
    radius_new = intermediate_radius + dot_radius_new * dt / 2

    return radius_new, dot_radius_new, dotdot_radius_new, dotdotdot_radius_new


def leapfrog_kdk_time_step(radius, dot_radius, dotdot_radius, dotdotdot_radius, mass_potential, dot_mass_potential, mass_gas, dot_mass_gas, dotdot_mass_gas, luminosity, dt):
    #kick half step
    intermediate_dotdotdot_radius = rtdot_calc(luminosity, mass_gas, dot_mass_gas, dotdot_mass_gas, mass_potential, dot_mass_potential, radius, dot_radius, dotdot_radius)
    intermediate_dot_radius = dot_radius + dotdot_radius * dt / 2
    intermediate_dot_radius, dotdot_radius, intermediate_dotdotdot_radius = cap_velocities(intermediate_dot_radius, dotdot_radius, intermediate_dotdotdot_radius)

    #drift full step
    dotdot_radius_new = dotdot_radius + intermediate_dotdotdot_radius * dt
    radius_new = radius + intermediate_dot_radius * dt

    #kick half step
    dotdotdot_radius_new = rtdot_calc(luminosity, mass_gas, dot_mass_gas, dotdot_mass_gas, mass_potential, dot_mass_potential, radius_new, intermediate_dot_radius, dotdot_radius_new)
    dot_radius_new = intermediate_dot_radius + dotdot_radius_new * dt / 2
    dot_radius_new, dotdot_radius_new, dotdotdot_radius_new = cap_velocities(dot_radius_new, dotdot_radius_new, dotdotdot_radius_new)

    return radius_new, dot_radius_new, dotdot_radius_new, dotdotdot_radius_new


def rtdot_calc(luminosity, mass_gas, mdot_gas, mddot_gas, mass_pot, mdot_pot, radius, rdot, rddot):
    #This is the equation of motion, based on equations 9-11 of Zubovas & King (2019), doi: 10.1093/mnras/stz105

    # Gravitational constant is 1 in code units; this is placed here for ease of
    # comparison with the equation of motion in the published paper.
    G = 1.0

    #Eq. 10:
    A_term = mdot_gas * rdot**2 + mass_gas * rdot * rddot + 2 * G * rdot / radius**2 * (mass_gas * mass_pot + mass_gas**2 / 2) #- G * (mdot_gas * mass_pot + mass_gas * mdot_pot + mass_gas * mdot_gas) / radius #this term might be actually required, considerations ongoing

    #Eq. 11:
    B_term = mddot_gas * rdot / mass_gas + mdot_gas * rdot**2 / (mass_gas * radius) + 2 * mdot_gas * rddot / mass_gas + rdot * rddot / radius + G * (mdot_gas * mass_pot + mass_gas * mdot_pot + mass_gas * mdot_gas) / (mass_gas * radius**2) - G * (2 * mass_gas * mass_pot * rdot + mass_gas**2 * rdot) / (2 * mass_gas * radius**3)

    #Eq. 9:
    rtdot = 3 * (const.GAMMA - 1) / (mass_gas * radius) * (const.ETA_DRIVE * luminosity - A_term) - B_term

    return rtdot


def cap_velocities(dot_radius, dotdot_radius, dotdotdot_radius):
    velocity_cap = 2 * const.ETA_DRIVE * const.c #outflow velocity cannot exceed driving wind velocity
    if dot_radius > velocity_cap:
        dot_radius = velocity_cap
        if dotdot_radius > 0:
            dotdot_radius = 0
        if dotdotdot_radius > 0:
            dotdotdot_radius = 0

    return dot_radius, dotdot_radius, dotdotdot_radius

