from sparseml.pytorch.utils.verbosity_helper import Verbosity
import pytest

@pytest.mark.parametrize("test_input,expected", [
    (1, Verbosity.DEFAULT),
    (2, Verbosity.ON_LR_CHANGE),
    (3, Verbosity.ON_EPOCH_CHANGE),
    (4, Verbosity.ON_LR_OR_EPOCH_CHANGE),
    (True, Verbosity.DEFAULT),
    (0, Verbosity.OFF),
    (False, Verbosity.OFF)
])
def test_convert_int_to_verbosity(test_input, expected):
    assert Verbosity.convert_int_to_verbosity(test_input) == expected


@pytest.mark.parametrize("test_input", [
    -1,
    float('inf'),
    "invalid_inp",
])
def test_exception(test_input):
    with pytest.raises(ValueError):
        assert Verbosity.convert_int_to_verbosity(test_input)
