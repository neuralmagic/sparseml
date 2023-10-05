from sparseml.core import State
from sparseml.core.framework import Framework
import pytest

@pytest.fixture
def state():
    # fixture to set up a state for each test
    #  that uses this fixture
    yield State(framework=Framework.pytorch)

def _dummy_model(*args, **kwargs):
    return 1

@pytest.mark.parametrize(
    "update_kwargs, expected_state", [
        ({"model": _dummy_model}, {"model": _dummy_model, "framework": Framework.pytorch}),
    ]
)
def test_update(state, update_kwargs, expected_state):
    original_state_dict = state.__dict__.copy()
    state.update(**update_kwargs)
    
    # check expected args are updated
    for key, value in expected_state.items():
        if key in ("model", "teacher_model"):
            assert getattr(state, key).model == value
        elif key == "optimizer":
            assert getattr(state, key).optimizer == value
        else:
            assert getattr(state, key) == value
    
    # check other args are not affected
    for key, value in original_state_dict.items():
        if key not in expected_state:
            assert getattr(state, key) == value
    