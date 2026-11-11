import magnofit.calc.time as tc
import magnofit.constants as const

import numpy as np
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
            0.0010000123352174365,
            0.006902170241785734,
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


# ---------------------------------------------------------------------------
# Tests of simple_time_step against independent, analytic expectations.
#
# The snapshot test above only guards against *unintended* changes. The tests
# below check that the scheme does what it is meant to do: given the third
# derivative of the radius evaluated at the start of the step, advance the
# second derivative, the first derivative and the radius itself with the Taylor
# expansion built from the values at the *start* of the step.
# ---------------------------------------------------------------------------

# Mass and luminosity arguments of simple_time_step; they are irrelevant
# whenever rtdot_calc is mocked out.
_MASS_AND_LUMINOSITY_ARGS = (2.0, 0.02, 1.0, 0.05, 0.001, 3.0)


def _step(radius, dot_radius, dotdot_radius, dt):
    return tc.simple_time_step(
        radius, dot_radius, dotdot_radius, 0.0, *_MASS_AND_LUMINOSITY_ARGS, dt
    )


@pytest.mark.parametrize(
    "jerk, radius, dot_radius, dotdot_radius, dt",
    [
        pytest.param(0.0, 1.0, 0.3, 0.05, 0.1, id="zero-jerk-constant-acceleration"),
        pytest.param(0.0, 1.0, 0.0, 0.0, 0.1, id="zero-jerk-at-rest"),
        pytest.param(2.0, 1.0, 0.3, 0.05, 0.1, id="positive-jerk"),
        pytest.param(-3.5, 2.0, -0.4, -0.2, 0.05, id="negative-jerk-and-motion"),
        pytest.param(40.0, 0.5, 1.2, 7.0, 0.01, id="large-jerk-small-step"),
        pytest.param(2.0, 1.0, 0.3, 0.05, 0.0, id="zero-time-step"),
    ],
)
def test_simple_time_step_exact_for_constant_jerk(
    monkeypatch, jerk, radius, dot_radius, dotdot_radius, dt
):
    # With a constant third derivative the Taylor expansion is exact, so every
    # output must match the closed-form kinematic expressions.
    monkeypatch.setattr(tc, "rtdot_calc", lambda *args, **kwargs: jerk)

    new_radius, new_dot_radius, new_dotdot_radius, new_dotdotdot_radius = _step(
        radius, dot_radius, dotdot_radius, dt
    )

    expected_dotdot_radius = dotdot_radius + jerk * dt
    expected_dot_radius = dot_radius + dotdot_radius * dt + 0.5 * jerk * dt**2
    expected_radius = (
        radius + dot_radius * dt + 0.5 * dotdot_radius * dt**2 + jerk * dt**3 / 6.0
    )

    assert new_dotdotdot_radius == pytest.approx(jerk, rel=1e-12, abs=1e-14)
    assert new_dotdot_radius == pytest.approx(
        expected_dotdot_radius, rel=1e-12, abs=1e-14
    )
    assert new_dot_radius == pytest.approx(expected_dot_radius, rel=1e-12, abs=1e-14)
    assert new_radius == pytest.approx(expected_radius, rel=1e-12, abs=1e-14)


def test_simple_time_step_evaluates_jerk_at_start_of_step(monkeypatch):
    # The third derivative must be computed from the state at the *start* of the
    # step, not from any partially updated quantity.
    recorded = {}

    def fake_rtdot_calc(
        luminosity,
        mass_gas,
        dot_mass_gas,
        dotdot_mass_gas,
        mass_potential,
        dot_mass_potential,
        radius,
        dot_radius,
        dotdot_radius,
    ):
        recorded["state"] = (radius, dot_radius, dotdot_radius)
        return 1.0

    monkeypatch.setattr(tc, "rtdot_calc", fake_rtdot_calc)
    _step(1.0, 0.3, 0.05, 0.1)
    assert recorded["state"] == (1.0, 0.3, 0.05)


def _frozen_masses_jerk(radius, dot_radius, dotdot_radius):
    # A closed third-order ODE for the radius: all mass terms and the luminosity
    # are held fixed, so that the physics reduces to d3r/dt3 = f(r, dr/dt, d2r/dt2).
    return tc.rtdot_calc(
        3.0, 1.0, 0.05, 0.001, 2.0, 0.02, radius, dot_radius, dotdot_radius
    )


def _reference_solution(initial_state, end_time, n_steps=200):
    """High-accuracy reference using classical RK4 on (r, dr/dt, d2r/dt2)."""

    def rhs(y):
        return np.array([y[1], y[2], _frozen_masses_jerk(*y)])

    y = np.array(initial_state, dtype=float)
    h = end_time / n_steps
    for _ in range(n_steps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * h * k1)
        k3 = rhs(y + 0.5 * h * k2)
        k4 = rhs(y + h * k3)
        y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return y


def test_simple_time_step_converges_to_reference_solution():
    # Integrate a full (non-constant-jerk) ODE with decreasing step sizes and
    # compare to an RK4 reference. The scheme evaluates the third derivative
    # only once per step, so the second derivative is advanced with a
    # forward-Euler step and the global error is expected to decrease linearly
    # with the step size.
    initial_state = (1.0, 0.3, 0.05)
    end_time = 0.2
    reference = _reference_solution(initial_state, end_time)
    # Verify that reference discretization error is far below the errors under
    # test. Doubling the RK4 resolution changes this state by less than 1e-12.
    np.testing.assert_allclose(
        reference,
        _reference_solution(initial_state, end_time, n_steps=400),
        rtol=0.0,
        atol=1e-12,
    )

    errors = []
    for n_steps in (40, 80, 160):
        dt = end_time / n_steps
        state = initial_state
        for _ in range(n_steps):
            state = tc.simple_time_step(
                *state, 0.0, 2.0, 0.02, 1.0, 0.05, 0.001, 3.0, dt
            )[:3]
        errors.append(np.abs(np.array(state) - reference))
    errors = np.array(errors)  # shape (3 resolutions, 3 variables)

    # Errors must shrink monotonically with the step size...
    assert np.all(errors[1] < errors[0])
    assert np.all(errors[2] < errors[1])

    # ...at (close to) first order: halving dt halves the error.
    observed_order = np.log2(errors[:-1] / errors[1:])
    assert np.all(observed_order > 0.8)
    assert np.all(observed_order < 1.3)

    # The absolute accuracy at the finest resolution is also pinned: the radius
    # error is ~3e-6 (relative) with the correct update formulas, but ~9e-5 if
    # the pre-step velocity/acceleration are replaced by their updated values.
    assert errors[-1][0] / reference[0] < 1e-5
