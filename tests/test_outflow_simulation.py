import astropy.table
import magnofit.calc.mass
import magnofit.constants as const
import numpy as np
import pytest
from magnofit.galaxy import Galaxy
from magnofit.simulation import run_outflow_simulation


def test_simulation_default_params():
    initial_galaxy_parameters = Galaxy()
    rng = np.random.default_rng(0)
    initial_galaxy_parameters.generate_stochastic_parameters(rng)

    outflow_properties = run_outflow_simulation(initial_galaxy_parameters, rng=None)
    expected_outflow_properties = astropy.table.Table.read(
        "tests/data/expected_outflow_properties.hdf5"
    )
    for col, expected_col in zip(
        outflow_properties.itercols(), expected_outflow_properties.itercols()
    ):
        assert np.allclose(col.data, expected_col.data)


def test_simulation_analytical():
    # Simulate an outflow in a purely isothermal matter distribution,
    # which allows for an analytical expression of velocity
    # (cf. eq. 12 of King, Zubovas & Power 2011, doi: 10.1111/j.1745-3933.2011.01067.x),
    # compare the result to that analytical expression;
    # note that the expression in that paper implicitly assumes f_g is very small;
    # in the expression used for this test, an extra factor (1-f_g/2) is added.

    initial_galaxy_parameters = Galaxy(
        bulge_mass=1e12 / const.UNIT_MSUN,
        bulge_sigma=2e7 / const.UNIT_VELOCITY,
        bulge_profile=magnofit.calc.mass.MassIsothermal(),
        bulge_gas_fraction=0.16,
        outflow_sphere_angle_ratio=1.0,
        duty_cycle=1.0,
        quasar_activity_duration=1.0e8 / const.UNIT_YEAR,
    )
    initial_galaxy_parameters.smbh_mass = (
        0.16
        * const._opacity
        / (np.pi * const.G ** 2)
        * pow(initial_galaxy_parameters.bulge_sigma * const.UNIT_VELOCITY, 4)
        / const.UNIT_MASS
    )

    outflow_properties = run_outflow_simulation(
        initial_galaxy_parameters, smbh_grows=False
    )
    outflow_properties.sort("time")
    calculated_velocity = (
        outflow_properties["dot_radius"][100:].mean() * 1.0e5
    )  # By step 100, velocity should definitely have reached an equlibrium value
    sigma_for_comparison = initial_galaxy_parameters.bulge_sigma * const.UNIT_VELOCITY
    fg_for_comparison = initial_galaxy_parameters.bulge_gas_fraction

    driving_term = 2 * (2 * const.ETA_DRIVE) * const.c * 0.16 / fg_for_comparison
    dynamics_term = (
        3 * calculated_velocity ** 3 / sigma_for_comparison ** 2
        + 10 * calculated_velocity * (1 - fg_for_comparison / 2)
    )

    assert driving_term ** (1.0 / 3) == pytest.approx((dynamics_term) ** (1.0 / 3))
