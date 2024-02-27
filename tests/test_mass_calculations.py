import magnofit.calc.mass as mc
from magnofit.galaxy import Galaxy
import numpy as np
import pytest


@pytest.mark.parametrize(
    "mass_calculator, radius, dot_radius, dotdot_radius,"
    "expected_mass_potential, expected_dot_mass_potential,"
    "expected_mass_gass, expected_dot_mass_gass, expected_dotdot_mass_gass,"
    "expected_rho_gas, expected_rho_gas_outer",
    [
        (
            mc.MassIsothermal(),
            0.011802848251257488,
            -1.2922067836067253,
            864.5197710937937,
            0.08089996952227271,
            -8.85714085149916,
            8.098095047274547e-05,
            -0.008866006858357518,
            5.931587975656372,
            6.467019679677523e-07,
            2.7292631329363096e-10,
        ),
        (
            mc.MassHernquist(),
            7.881973653326038,
            2.9526692774655885,
            0.1079021152905847,
            9.068140060216543,
            5.344313238297087,
            0.009077217277494037,
            0.005349662901198285,
            0.0009166590559184575,
            2.320768332159257e-06,
            1.4168607554632548e-09,
        ),
        (
            mc.MassJaffe(),
            0.023082754953723632,
            2.1997299938986394,
            -62.8301873701011,
            0.1580899649696047,
            15.053627532528292,
            0.00015824821318278747,
            0.015068696228757051,
            -0.42812261190105466,
            1.0231072333551823,
            0.0005751933926857654,
        ),
        (
            mc.MassNFW(),
            4.838925618598275,
            3.0882881119106758,
            0.6130920237344336,
            1.5086170208380532,
            1.7400167060537297,
            0.0015101271479860392,
            0.001741758464518248,
            0.0011400054103497587,
            3.5493991637961736e-06,
            2.921435680648168e-09,
        ),
    ],
)
def test_mass_calculations(
    mass_calculator,
    radius,
    dot_radius,
    dotdot_radius,
    expected_mass_potential,
    expected_dot_mass_potential,
    expected_mass_gass,
    expected_dot_mass_gass,
    expected_dotdot_mass_gass,
    expected_rho_gas,
    expected_rho_gas_outer,
):
    rng = np.random.default_rng(42)
    init_params = Galaxy(halo_profile=mass_calculator)
    init_params.generate_stochastic_parameters(rng)

    (
        mass_potential,
        dot_mass_potential,
        mass_gass,
        dot_mass_gass,
        dotdot_mass_gass,
        rho_gas,
        rho_gas_outer,
    ) = init_params.halo_profile.calculate(
        radius,
        dot_radius,
        dotdot_radius,
        init_params.virial_mass * (1.0 - init_params.bulge_to_total_mass_fraction),
        init_params.virial_radius / init_params.halo_concentration,
        init_params.halo_concentration,
        init_params.halo_gas_fraction,
    )

    assert mass_potential == pytest.approx(expected_mass_potential)
    assert dot_mass_potential == pytest.approx(expected_dot_mass_potential)
    assert mass_gass == pytest.approx(expected_mass_gass)
    assert dot_mass_gass == pytest.approx(expected_dot_mass_gass)
    assert dotdot_mass_gass == pytest.approx(expected_dotdot_mass_gass)
    assert rho_gas == pytest.approx(expected_rho_gas)
    assert rho_gas_outer == pytest.approx(expected_rho_gas_outer)
