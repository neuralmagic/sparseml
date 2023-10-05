from sparseml.core.framework import Framework
import sparseml.core.session as session_module
import pytest

@pytest.fixture
def setup_session():
    # fixture to set up a session for each test
    #  that uses this fixture
    session_module.create_session()
    yield 
    
@pytest.fixture
def setup_active_session(setup_session):
    # fixture to set up an active session for each test
    #  that uses this fixture
    yield session_module.active_session()
    

@pytest.mark.parametrize(
    "attribute_name", [
        "active_session",
        "create_session",
    ]
    )
def test_session_has_base_attribute(attribute_name):
    # tests to cover name changes and 
    #  backwards compatibility
    assert hasattr(session_module, attribute_name), f"session does not have {attribute_name} attribute"

def test_active_session_without_create_session():
    actual_session = session_module.active_session()
    assert actual_session

def test_active_session_returns_sparse_session(setup_session):
    actual_session = session_module.active_session()
    assert isinstance(actual_session, session_module.SparseSession), "create_session did not return a SparseSession"

def test_active_session_returns_same_session_on_subsequent_calls(setup_session):
    actual_session = session_module.active_session()
    assert actual_session is session_module.active_session(), "active_session did not return the same session"

def test_active_session_returns_created_session(setup_session):
    actual_session = session_module.active_session()
    assert actual_session is session_module._global_session, "active_session did not return the created session"

def test_create_session_yields_new_sessions():
    session_a = session_module.create_session()
    session_b = session_module.create_session()
    
    assert isinstance(session_a, type(session_b)), "create_session did not return the same type of session"
    assert session_a is not session_b, "create_session did not return a new session"


def _get_valid_recipes():
    recipes = [None, "_test_recipe_mobilenet_v2.yaml"]
    return recipes
    
    
@pytest.mark.parametrize("framework", [framework.value for framework in Framework])
@pytest.mark.parametrize("recipe", _get_valid_recipes())
def test_initialize_returns_modified_state(framework, recipe):
    result = session_module.initialize(framework=framework, recipe=recipe)
    assert isinstance(result, session_module.ModifiedState), "initialize did not return a ModifiedState"
    
@pytest.mark.xfail(reason="initialize does not accept empty arguments, and needs the framework argument at minimum")
def test_initialize_accepts_empty_arguments():
    result = session_module.initialize()
    assert isinstance(result, session_module.ModifiedState), "initialize did not return a ModifiedState"



def test_initialize_can_be_called_multiple_times_to_set_state(setup_session):
    session_module.initialize(framework=Framework.pytorch)
    state = session_module.active_session().lifecycle.state
    expected_unset_fields = [
    "model",
    "teacher_model",
    "optimizer",
    "optim_wrapped",
    "loss",
    "batch_data",
    "last_event",
    "start_event",
    ]
    
    assert all(hasattr(state, field) and getattr(state, field) is None for field in expected_unset_fields), f"initialize did not set some fields to None"
    
    session_module.initialize(start=1.0)
    new_state = session_module.active_session().lifecycle.state
    
    # check if start_event is set
    assert new_state.start_event and new_state.start_event.current_index == 1.0, "initialize did not set start_event"
    
    # check previous state was not overwritten
    assert new_state.framework and new_state.framework == Framework.pytorch, "initialize overwrote previous state"
    
    # check other fields were not affected
    
    # remove start_event from expected_unset_fields
    expected_unset_fields.pop(-1)
    
    assert all(hasattr(state, field) and getattr(state, field) is None for field in expected_unset_fields), "initialize set wrong state fields"
    