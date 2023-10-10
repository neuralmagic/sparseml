
from collections import defaultdict
from types import SimpleNamespace
import pytest
import os
from sparseml.core.framework import Framework
import sparseml.core.session as session_module
from tests.sparseml.helpers import valid_recipe_strings




class LifeCycleMock():
    """
    Mock class to track lifecycle method calls
    """
    
    def __init__(self, model=None, optimizer=None, loss=None):
        self._state = SimpleNamespace(
            model=SimpleNamespace(model=model),
            optimizer=SimpleNamespace(optimizer=optimizer),
            loss=SimpleNamespace(loss=loss),
            )
        self._hit_count = defaultdict(int)

    def _increase_hit_count(self, method_name):
        self._hit_count[method_name] += 1
        
    def pre_initialize_structure(self, *args, **kwargs):
        self._increase_hit_count("pre_initialize_structure")
        return "pre_initialize_structure"
    
    def initialize(self, *args, **kwargs):
        self._increase_hit_count("initialize")
        return "initialize"
    
    def finalize(self, *args, **kwargs):
        self._increase_hit_count("finalize")
        return "finalize"
    
    
    def event(self, *args, **kwargs):
        self._increase_hit_count("event")
        return "event"
    
    def reset(self, *args, **kwargs):
        self._increase_hit_count("reset")
        
    
    @property
    def state(self):
        return self._state
 

def get_linear_net():
    from tests.sparseml.pytorch.helpers import LinearNet
    return LinearNet()

class TestSparseSession:
    def test_session_has_a_sparsification_lifecycle(self, setup_active_session):
        assert hasattr(setup_active_session, "lifecycle"), "SparseSession does not have a lifecyle"
        
        lifecyle = setup_active_session.lifecycle
        assert isinstance(lifecyle, session_module.SparsificationLifecycle), "SparseSession.lifecycle is not a SparsificationLifecycle"


    @pytest.mark.skipif(
        os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
        reason="Skipping pytorch tests",
    )
    def test_initialize_can_be_called_multiple_times_to_set_state(self, setup_session):
        session_module.initialize(framework=Framework.pytorch)
        state = session_module.active_session().lifecycle.state

        assert state.model is None
        model = get_linear_net()
        session_module.initialize(model=model)
        
        import torch
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        session_module.initialize(optimizer=optimizer)
        
        # assert model was not overwritten
        assert state.model.model is model


    @pytest.mark.parametrize("method_name, kwargs", [
        ("pre_initialize_structure", {"model": get_linear_net(), "framework": Framework.pytorch}),
        ("initialize", {"framework": Framework.pytorch}),
        ("finalize", {}),
        ("event", {"event_type": "test"}),
        ("reset", {}),
    ])
    def test_session_methods_invoke_lifecycle_methods(self, method_name, kwargs, monkeypatch, setup_active_session):
        
        monkeypatch.setattr(setup_active_session, "_lifecycle" , lifecycle_mock := LifeCycleMock(model=kwargs.get("model")))
        method = getattr(setup_active_session, method_name)
        
        result = method(**kwargs)
        if method_name != "reset":
            assert result.modifier_data == method_name, f"{method_name} did not invoke the lifecycle method"
        else:
            assert lifecycle_mock._hit_count[method_name] == 1, f"{method_name} did not invoke the lifecycle method"

    def test_apply_calls_lifecycle_initialize_and_finalize(self, setup_active_session, monkeypatch):
        monkeypatch.setattr(setup_active_session, "_lifecycle" , lifecycle_mock := LifeCycleMock())
        setup_active_session.apply()
        
        # check initialize was called once
        assert lifecycle_mock._hit_count["initialize"] == 1, "apply did not invoke the lifecycle initialize method"
        
        # check finalize was called once
        assert lifecycle_mock._hit_count["finalize"] == 1, "apply did not invoke the lifecycle finalize method"
        

@pytest.mark.parametrize("attribute_name", [
    "create_session",
    "active_session",
    "pre_initialize_structure",
    "initialize",
    "finalize",
    "apply",
])
def test_import(attribute_name):
    # this test will fail if the attribute is not found
    #  and will serve as a reminder to update the usages
    #  if the attribute is renamed or removed
    
    assert hasattr(session_module, attribute_name), f"{attribute_name} not found in sparseml.core.session"



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
    

def test_active_session_returns_sparse_session(setup_active_session):
    assert isinstance(setup_active_session, session_module.SparseSession), "create_session did not return a SparseSession"

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

def test_create_session_yields_new_sessions(setup_active_session):
    session_a = setup_active_session
    with session_module.create_session() as session_b:
        assert isinstance(session_b, type(session_a)), "create_session did not return the same type of session"
        assert session_a is not session_b, "create_session did not return a new session"

@pytest.mark.parametrize("framework", [framework for framework in Framework])
def test_initialize_returns_modified_state(framework):
    result = session_module.initialize(framework=framework)
    assert isinstance(result, session_module.ModifiedState), "initialize did not return a ModifiedState"

@pytest.mark.parametrize("method_name", [
    "pre_initialize_structure",
    "initialize",
    "finalize",
    ])
def test_module_methods_call_session_methods(method_name, monkeypatch):
    session_mock = LifeCycleMock()
    def active_session_mock():
        return session_mock
    
    monkeypatch.setattr(session_module, "active_session", active_session_mock)
    method = getattr(session_module, method_name)
    result = method()
    assert session_mock._hit_count[method_name] == 1, f"{method_name} did not invoke equivalent session method"
    if result is not None:
        assert result == method_name, f"{method_name} did not return the result of the equivalent session method"
