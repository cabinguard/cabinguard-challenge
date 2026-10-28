import math
import pytest

from src.cybersecurity.secure_message import MessageValidator


@pytest.mark.parametrize(
    "invalid",
    [-1, float("inf"), float("-inf"), float("nan"), True, "5"],
)
def test_invalid_max_age_rejected(invalid):
    with pytest.raises(ValueError):
        MessageValidator({}, {}, max_age_s=invalid)


@pytest.mark.parametrize(
    "invalid",
    [-1, float("inf"), float("-inf"), float("nan"), True, "2"],
)
def test_invalid_future_skew_rejected(invalid):
    with pytest.raises(ValueError):
        MessageValidator(
            {}, {},
            max_age_s=5,
            max_future_skew_s=invalid,
        )


def test_valid_configuration():
    validator = MessageValidator(
        {}, {},
        max_age_s=5,
        max_future_skew_s=2,
    )

    assert validator.max_age == 5.0
    assert validator.max_future == 2.0


def test_default_future_skew():
    validator = MessageValidator({}, {}, max_age_s=5)

    assert validator.max_future == 5.0


# --------------------------------------------------------------------------------- additional coverage
@pytest.mark.parametrize("invalid", [[], {}, object()])
def test_invalid_max_age_type_rejected(invalid):
    with pytest.raises(ValueError):
        MessageValidator({}, {}, max_age_s=invalid)


def test_max_age_false_rejected():
    """False == 0 numerically but must still be rejected: booleans are never acceptable in a
    numeric configuration field, matching the same rule enforced on message seq/ts/payload."""
    with pytest.raises(ValueError):
        MessageValidator({}, {}, max_age_s=False)


def test_future_skew_false_rejected():
    with pytest.raises(ValueError):
        MessageValidator({}, {}, max_age_s=5, max_future_skew_s=False)


@pytest.mark.parametrize("invalid", [[], {}, object()])
def test_invalid_future_skew_type_rejected(invalid):
    with pytest.raises(ValueError):
        MessageValidator({}, {}, max_age_s=5, max_future_skew_s=invalid)


def test_zero_max_age_is_a_valid_boundary_not_rejected():
    """Zero is a strict-but-legitimate tolerance (no slack at all) and must not be treated as
    falsy/invalid - distinct from the 'invalid' parametrize lists above, which only ever
    include negative values, never zero."""
    validator = MessageValidator({}, {}, max_age_s=0.0)
    assert validator.max_age == 0.0
    assert validator.max_future == 0.0  # default skew mirrors max_age


def test_zero_future_skew_is_a_valid_boundary_not_rejected():
    validator = MessageValidator({}, {}, max_age_s=5.0, max_future_skew_s=0.0)
    assert validator.max_future == 0.0
    assert validator.max_age == 5.0


def test_very_small_negative_max_age_still_rejected():
    """Boundary check just below zero - must not be accidentally accepted by a loose
    tolerance/epsilon comparison."""
    with pytest.raises(ValueError):
        MessageValidator({}, {}, max_age_s=-1e-12)


def test_very_small_negative_future_skew_still_rejected():
    with pytest.raises(ValueError):
        MessageValidator({}, {}, max_age_s=5, max_future_skew_s=-1e-12)


def test_integer_max_age_is_normalised_to_float():
    validator = MessageValidator({}, {}, max_age_s=5)
    assert validator.max_age == 5.0
    assert isinstance(validator.max_age, float)


def test_float_max_age_and_skew_preserved():
    validator = MessageValidator({}, {}, max_age_s=5.25, max_future_skew_s=1.75)
    assert validator.max_age == 5.25
    assert validator.max_future == 1.75


def test_large_finite_values_accepted():
    """Finite but very large tolerances are legitimate configuration choices (e.g. a lab/replay
    testbed with a deliberately huge age tolerance) and must not be rejected just for being big,
    only for being non-finite."""
    validator = MessageValidator({}, {}, max_age_s=1e9, max_future_skew_s=1e9)
    assert validator.max_age == 1e9
    assert validator.max_future == 1e9


def test_future_skew_none_means_default_to_max_age_not_an_error():
    """Explicitly passing max_future_skew_s=None is the documented 'use the default' signal, and
    must be distinguished from an actually-invalid value - must not raise."""
    validator = MessageValidator({}, {}, max_age_s=3.0, max_future_skew_s=None)
    assert validator.max_future == 3.0