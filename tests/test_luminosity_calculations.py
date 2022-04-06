import pytest

from magnofit.galaxy import Galaxy

import magnofit.calc.luminosity as lc


@pytest.mark.parametrize(
    "fade, time, expected_coef",
    [
        # fmt: off
        (lc.LuminosityFadeNone(),        0.1,   0.0                  ),
        (lc.LuminosityFadeNone(),        0.001, 1.0                  ),
        (lc.LuminosityFadeExponential(), 0.1,   0.0012791711490824902),
        (lc.LuminosityFadeExponential(), 0.001, 0.011553880624562867 ),
        (lc.LuminosityFadePowerLaw(),    0.1,   0.00273797247686821  ),
        (lc.LuminosityFadePowerLaw(),    0.001, 0.027379724768682098 ),
        (lc.LuminosityFadeKing(),        0.1,   0.0004719677682289763),
        (lc.LuminosityFadeKing(),        0.001, 0.0941659644495734   ),
        # fmt: on
    ],
)
def test_luminosity_coefficient(fade, time, expected_coef):
    initial_galaxy_parameters = Galaxy(fade=fade)

    coef = fade.luminosity_coefficient(time, initial_galaxy_parameters)
    assert coef == pytest.approx(expected_coef)


@pytest.mark.parametrize(
    "fade, time, dt1, dt2, expected_coef",
    [
        # fmt: off
        (lc.LuminosityFadeNone(),        0.1,   1e-6, 1e-6,  1.0                   ),
        (lc.LuminosityFadeNone(),        0.001, 1e-6, 1e-6,  1.0                   ),
        (lc.LuminosityFadeNone(),        0.1,   1e-5, 1e-6,  10.0                  ),
        (lc.LuminosityFadeNone(),        0.001, 1e-5, 1e-6,  10.0                  ),
        (lc.LuminosityFadeExponential(), 0.1,   1e-6, 1e-6,  0.0012791569309479355 ),
        (lc.LuminosityFadeExponential(), 0.001, 1e-6, 1e-6,  0.011553752201849874  ),
        (lc.LuminosityFadeExponential(), 0.1,   1e-5, 1e-6,  0.012790289772224706  ),
        (lc.LuminosityFadeExponential(), 0.001, 1e-5, 1e-6,  0.11552596483109227   ),
        (lc.LuminosityFadePowerLaw(),    0.1,   1e-6, 1e-6, -0.0027379656319571795 ),
        (lc.LuminosityFadePowerLaw(),    0.001, 1e-6, 1e-6, -0.027372883257814472  ),
        (lc.LuminosityFadePowerLaw(),    0.1,   1e-5, 1e-6, -0.02737904030975659   ),
        (lc.LuminosityFadePowerLaw(),    0.001, 1e-5, 1e-6, -0.2731161557914118    ),
        (lc.LuminosityFadeKing(),        0.1,   1e-6, 1e-6,  0.0004719649703965489 ),
        (lc.LuminosityFadeKing(),        0.001, 1e-6, 1e-6,  0.09411772931041068   ),
        (lc.LuminosityFadeKing(),        0.1,   1e-5, 1e-6,  0.004719397915026842  ),
        (lc.LuminosityFadeKing(),        0.001, 1e-5, 1e-6,  0.9368632681259501    ),
        # fmt: on
    ],
)
def test_mean_luminosity_coefficient(fade, time, dt1, dt2, expected_coef):
    initial_galaxy_parameters = Galaxy(fade=fade)

    mean_coef = fade.luminosity_mean_coefficient(
        time, dt1, dt2, initial_galaxy_parameters
    )
    assert mean_coef == pytest.approx(expected_coef)


@pytest.mark.parametrize(
    "fade, expected_quasar_lum_variation_ts",
    [
        # fmt: off
        (lc.LuminosityFadeNone(),         0.007497242933416249  ),
        (lc.LuminosityFadeExponential(), -0.19965923487105416   ),
        (lc.LuminosityFadePowerLaw(),     7.49649328408784e-07  ),
        (lc.LuminosityFadeKing(),         0.00015840594632720974),
        # fmt: on
    ],
)
def test_episode_quasar_lum_variation_ts(fade, expected_quasar_lum_variation_ts):
    initial_galaxy_parameters = Galaxy(fade=fade)

    quasar_lum_variation_ts = fade.quasar_luminosity_variation_timescale(
        initial_galaxy_parameters
    )
    assert quasar_lum_variation_ts == pytest.approx(expected_quasar_lum_variation_ts)
