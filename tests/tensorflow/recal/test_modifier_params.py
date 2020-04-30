import pytest

from typing import Callable
import numpy

from neuralmagicML.utils import ALL_TOKEN
from neuralmagicML.tensorflow.utils import (
    tf_compat,
    batch_cross_entropy_loss,
)
from neuralmagicML.tensorflow.recal import (
    TrainableParamsModifier,
    ScheduledModifierManager,
    EXTRAS_KEY_VAR_LIST,
)
from tests.tensorflow.helpers import mlp_net
from tests.tensorflow.recal.test_modifier import (
    ScheduledModifierTest,
    mlp_graph_lambda,
    conv_graph_lambda,
)


@pytest.mark.parametrize(
    "graph_lambda,modifier_lambda",
    [
        (
            mlp_graph_lambda,
            lambda: TrainableParamsModifier(
                params=["weight"],
                layers=["mlp_net/fc1/matmul"],
                trainable=False,
                params_strict=True,
            ),
        ),
        (
            mlp_graph_lambda,
            lambda: TrainableParamsModifier(
                params=ALL_TOKEN,
                layers=ALL_TOKEN,
                trainable=False,
                params_strict=False,
            ),
        ),
        (
            conv_graph_lambda,
            lambda: TrainableParamsModifier(
                params=["bias"],
                layers=["conv_net/conv1/add"],
                trainable=False,
                params_strict=True,
            ),
        ),
        (
            conv_graph_lambda,
            lambda: TrainableParamsModifier(
                params=["weight", "bias"],
                layers=ALL_TOKEN,
                trainable=False,
                params_strict=False,
            ),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestTrainableParamsModifierImpl(ScheduledModifierTest):
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], TrainableParamsModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()
        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()

            _, mod_extras = modifier.create_ops(steps_per_epoch, global_step, graph)
            assert len(mod_extras) == 1
            assert EXTRAS_KEY_VAR_LIST in mod_extras
            var_list = mod_extras[EXTRAS_KEY_VAR_LIST]
            assert len(var_list) > 0
            for var in var_list:
                if modifier._trainable:
                    assert var in tf_compat.trainable_variables()
                else:
                    assert var not in tf_compat.trainable_variables()

            with tf_compat.Session(graph=graph) as sess:
                # No ops to invoke

                modifier.complete_graph(graph, sess)
                for var, is_var_trainable in modifier._vars_to_trainable_orig.items():
                    if is_var_trainable:
                        assert var in tf_compat.trainable_variables()
                    else:
                        assert var not in tf_compat.trainable_variables()


def test_trainable_params_modifier_with_training():
    modifier = TrainableParamsModifier(
        params=["weight"],
        layers=["mlp_net/fc1/matmul"],
        trainable=False,
        params_strict=False,
    )
    manager = ScheduledModifierManager([modifier])
    steps_per_epoch = 5
    batch_size = 2

    with tf_compat.Graph().as_default() as graph:
        logits, inputs = mlp_net()
        labels = tf_compat.placeholder(tf_compat.float32, [None, *logits.shape[1:]])
        loss = batch_cross_entropy_loss(logits, labels)

        global_step = tf_compat.train.get_or_create_global_step()
        num_trainable_variabls_init = len(tf_compat.trainable_variables())

        mod_ops, mod_extras = manager.create_ops(steps_per_epoch)
        assert len(tf_compat.trainable_variables()) < num_trainable_variabls_init

        non_trainable_var = mod_extras[EXTRAS_KEY_VAR_LIST][0]
        trainable_var = tf_compat.trainable_variables()[0]

        train_op = tf_compat.train.AdamOptimizer(learning_rate=1e-4).minimize(
            loss, global_step=global_step
        )

        with tf_compat.Session(graph=graph) as sess:
            sess.run(tf_compat.global_variables_initializer())
            manager.initialize_session(sess)
            non_trainable_var_value_init = non_trainable_var.eval(session=sess)
            trainable_var_value_init = trainable_var.eval(session=sess)
            batch_lab = numpy.random.random((batch_size, *logits.shape[1:]))
            batch_inp = numpy.random.random((batch_size, *inputs.shape[1:]))

            for epoch in range(10):
                for step in range(steps_per_epoch):
                    sess.run(train_op, feed_dict={inputs: batch_inp, labels: batch_lab})
                    step_counter = sess.run(global_step)

            non_trainable_var_value_final = non_trainable_var.eval(session=sess)
            trainable_var_value_final = trainable_var.eval(session=sess)
            assert numpy.array_equal(
                non_trainable_var_value_init, non_trainable_var_value_final
            )
            assert not numpy.array_equal(
                trainable_var_value_init, trainable_var_value_final
            )
            manager.complete_graph()


def test_trainable_params_yaml():
    params = ALL_TOKEN
    layers = ALL_TOKEN
    trainable = False
    params_strict = False
    yaml_str = f"""
    !TrainableParamsModifier
        params: {params}
        layers: {layers}
        trainable: {trainable}
        params_strict: {params_strict}
    """
    yaml_modifier = TrainableParamsModifier.load_obj(
        yaml_str
    )  # type: TrainableParamsModifier
    serialized_modifier = TrainableParamsModifier.load_obj(
        str(yaml_modifier)
    )  # type: TrainableParamsModifier
    obj_modifier = TrainableParamsModifier(
        params=params, layers=layers, trainable=trainable, params_strict=params_strict
    )

    assert isinstance(yaml_modifier, TrainableParamsModifier)
    assert yaml_modifier.params == serialized_modifier.params == obj_modifier.params
    assert yaml_modifier.layers == serialized_modifier.layers == obj_modifier.layers
    assert (
        yaml_modifier.trainable
        == serialized_modifier.trainable
        == obj_modifier.trainable
    )
    assert (
        yaml_modifier.params_strict
        == serialized_modifier.params_strict
        == obj_modifier.params_strict
    )
