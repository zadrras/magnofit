import pytest
import numpy as np

from magnofit.galaxy import Galaxy
import magnofit.constants as const


@pytest.mark.parametrize(
    "rng, expected_bulge_mass",
    [
        (np.random.RandomState(0), 2714531156),
        (np.random.RandomState(42), 2840831965),
    ],
)
def test_bulge_mass(rng, expected_bulge_mass):
    initial_galaxy_parameters = Galaxy()
    initial_galaxy_parameters.generate_stochastic_parameters(rng)

    assert initial_galaxy_parameters.bulge_mass * const.UNIT_MSUN == pytest.approx(expected_bulge_mass)


@pytest.mark.parametrize(
    "rng, expected_smbh_mass",
    [
        (np.random.RandomState(0), 4667139),
        (np.random.RandomState(42), 3385572),
    ],
)
def test_smbh_mass(rng, expected_smbh_mass):
    initial_galaxy_parameters = Galaxy()
    initial_galaxy_parameters.generate_stochastic_parameters(rng)

    assert initial_galaxy_parameters.smbh_mass * const.UNIT_MSUN == pytest.approx(expected_smbh_mass)

@pytest.mark.parametrize(
    "rng, expected_bulge_sigma",
    [
        (np.random.RandomState(0), 10523399),
        (np.random.RandomState(42), 10347858),
    ],
)
def test_bulge_sigma(rng, expected_bulge_sigma):
    initial_galaxy_parameters = Galaxy()
    initial_galaxy_parameters.generate_stochastic_parameters(rng)

    assert initial_galaxy_parameters.bulge_sigma * const.UNIT_VELOCITY == pytest.approx(expected_bulge_sigma)



