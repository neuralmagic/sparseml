import pytest
from sparseml.core.framework import Framework
import sparseml.core.session as sml

def recipe_with_layer_prefix():
    layer_prefix = "decoder"
    recipe = f"""
    metadata:
        target_model:
            layer_prefix: {layer_prefix}
            architecture: "opt"

    test_stage:
        pruning_modifiers:
            ConstantPruningModifier:
                targets: __ALL_PRUNABLE__
                start: 0
                end: 5
    """
    return recipe, layer_prefix

def recipe_without_layer_prefix():
    recipe = """
    metadata:
        target_model:
            architecture: "opt"

    test_stage:
        pruning_modifiers:
            ConstantPruningModifier:
                targets: __ALL_PRUNABLE__
                start: 0
                end: 5
    """
    return recipe, None

@pytest.fixture
def model():
    # identity model
    return lambda x: x


@pytest.mark.parametrize(
    "recipe, expected_layer_prefix",
    [
        recipe_with_layer_prefix(),
        recipe_without_layer_prefix(), # layer prefix should be none
    ],
)
def test_session_initialize_propagates_layer_prefix_to_model(recipe, expected_layer_prefix, model):
    session = sml.active_session()
    session.initialize(framework=Framework.general ,model=model, recipe=recipe)
    assert session.state.model.layer_prefix == expected_layer_prefix