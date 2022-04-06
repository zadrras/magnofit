import magnofit.calc.time as tc
import magnofit.constants as const

import pytest


@pytest.mark.parametrize(
    "integration_method, radius, dot_radius, dotdot_radius, dotdotdot_radius,"
    "mass_potential, dot_mass_potential, mass_gas, dot_mass_gas, dotdot_mass_gas,"
    "mean_luminosity, dt,"
    "expected_radius, expected_dot_radius, expected_dotdot_radius, expected_dotdotdot_radius",
    [
        (
            tc.simple_time_step,
            0.0010000113011407227,
            0.0068907547333344745,
            72.82782012531469,
            44078612.115042835,
            0.0011437245036460388,
            0.007881691689981915,
            0.00027234309730229993,
            0.0018766289351604545,
            19.83393690287684,
            120.13386279763026,
            1.4994485866832498e-07,
            0.0010000123371519593,
            0.006903160944296629,
            79.43493236439795,
            44063613.1025877,
        ),
        (
            tc.leapfrog_dkd_time_step,
            0.001000935666615366,
            0.06422908322735521,
            3163.7875063864735,
            84833383.46965575,
            0.0010318971208565972,
            0.06622273974301864,
            0.00012915369035278738,
            0.008287675528473468,
            408.23320267552094,
            111.58277609194681,
            4.0602653052837037e-07,
            0.0010009620075339194,
            0.06552065760350481,
            3198.212727905005,
            84737907.7153473,
        ),
        (
            tc.leapfrog_kdk_time_step,
            0.0014818143211041733,
            3.7277742585431453,
            5138.997854875078,
            -94501353.53508157,
            0.006129035135173488,
            15.419718190967664,
            0.0016769683027169986,
            4.218720715251636,
            5815.807021751587,
            2339.0883677149013,
            1.087603015753068e-06,
            0.0014858716686454276,
            3.7333075562124067,
            5036.624337696801,
            -92678995.67979833,
        ),
    ],
)
def test_time_calculations(
    integration_method,
    radius,
    dot_radius,
    dotdot_radius,
    dotdotdot_radius,
    mass_potential,
    dot_mass_potential,
    mass_gas,
    dot_mass_gas,
    dotdot_mass_gas,
    mean_luminosity,
    dt,
    expected_radius,
    expected_dot_radius,
    expected_dotdot_radius,
    expected_dotdotdot_radius,
):
    new_radius, new_dot_radius, new_dotdot_radius, new_dotdotdot_radius = integration_method(
        radius,
        dot_radius,
        dotdot_radius,
        dotdotdot_radius,
        mass_potential,
        dot_mass_potential,
        mass_gas,
        dot_mass_gas,
        dotdot_mass_gas,
        mean_luminosity,
        dt,
    )

    assert new_radius == pytest.approx(expected_radius)
    assert new_dot_radius == pytest.approx(expected_dot_radius)
    assert new_dotdot_radius == pytest.approx(expected_dotdot_radius)
    assert new_dotdotdot_radius == pytest.approx(expected_dotdotdot_radius)
