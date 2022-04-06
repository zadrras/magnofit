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
            0.08092930187774189,
            -8.86035223471019,
            8.101031218993182e-05,
            -0.008869221456166356,
            5.933738624760766,
            6.469364463256437e-07,
            2.7292631329363096e-10,
        ),
        (
            mc.MassHernquist(),
            7.881973653326038,
            2.9526692774655885,
            0.1079021152905847,
            9.071427946593523,
            5.346250955907736,
            0.009080508455048571,
            0.005351602558466202,
            0.0009169914141310872,
            2.3216097861435103e-06,
            1.4168607554632548e-09,
        ),
        (
            mc.MassJaffe(),
            0.023082754953723632,
            2.1997299938986394,
            -62.8301873701011,
            0.15814728453444482,
            15.059085610652327,
            0.00015830559012456938,
            0.01507415977042275,
            -0.42827783871647657,
            1.0234781870802463,
            0.0005754019437786448,
        ),
        (
            mc.MassNFW(),
            4.838925618598275,
            3.0882881119106758,
            0.6130920237344336,
            1.509164008568498,
            1.7406475936652537,
            0.00151067468325175,
            0.0017423899836489026,
            0.0011404187485021785,
            3.5506860893486705e-06,
            2.9224949219602616e-09,
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
    rng = np.random.RandomState(42)
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
