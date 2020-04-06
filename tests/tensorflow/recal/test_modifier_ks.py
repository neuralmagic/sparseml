import pytest

from typing import Callable
import numpy

from neuralmagicML.tensorflow.utils import VAR_INDEX_FROM_TRAINABLE, tf_compat
from neuralmagicML.tensorflow.recal import GradualKSModifier

from tests.tensorflow.helpers import mlp_net
from tests.tensorflow.recal.test_modifier import ScheduledModifierTest


@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: GradualKSModifier(
            layers=["mlp_net/fc1/matmul"],
            init_sparsity=0.05,
            final_sparsity=0.8,
            start_epoch=0.0,
            end_epoch=20.0,
            update_frequency=1.0,
        ),
        lambda: GradualKSModifier(
            layers="__ALL__",
            init_sparsity=0.05,
            final_sparsity=0.8,
            start_epoch=0.0,
            end_epoch=20.0,
            update_frequency=1.0,
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("graph_lambda", [mlp_net], scope="function")
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestGradualKSModifierImpl(ScheduledModifierTest):
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], GradualKSModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()
        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()
            step_placeholder = tf_compat.placeholder(dtype=tf_compat.int64, name="step")
            global_assign = global_step.assign(step_placeholder)

        graph, ops = modifier.create_ops(graph, steps_per_epoch, global_step)
        assert len(ops) == 1
        assert ops[0] is not None
        assert modifier.prune_op_vars
        assert len(modifier.prune_op_vars) > 0
        last_sparsities = [0.0 for _ in range(len(modifier.prune_op_vars))]

        with tf_compat.Session(graph=graph) as sess:
            sess.run(tf_compat.global_variables_initializer())
            step_counter = 0

            for epoch in range(int(modifier.end_epoch + 5.0)):
                for step in range(steps_per_epoch):
                    step_counter += 1
                    sess.run(global_assign, feed_dict={step_placeholder: step_counter})
                    sess.run(ops[0])

                    for index, op_vars in enumerate(modifier.prune_op_vars):
                        mask_val = sess.run(op_vars.mask)
                        num_nonzeros = numpy.count_nonzero(mask_val)
                        calc_density = float(num_nonzeros) / float(mask_val.size)
                        calc_sparsity = 1.0 - calc_density

                        if epoch < modifier.start_epoch:
                            assert calc_sparsity == 0.0
                        elif epoch > modifier.end_epoch:
                            assert abs(calc_sparsity - modifier.final_sparsity) < 1e-2
                        else:
                            assert calc_sparsity >= last_sparsities[index]
                            last_sparsities[index] = calc_sparsity


def test_gradual_ks_yaml():
    layers = "__ALL__"
    init_sparsity = 0.05
    final_sparsity = 0.8
    start_epoch = 5.0
    end_epoch = 15.0
    update_frequency = 1.0
    param = VAR_INDEX_FROM_TRAINABLE
    inter_func = "cubic"
    yaml_str = f"""
    !GradualKSModifier
        layers: {layers}
        init_sparsity: {init_sparsity}
        final_sparsity: {final_sparsity}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
        param: {param}
        inter_func: {inter_func}
    """
    yaml_modifier = GradualKSModifier.load_obj(yaml_str)  # type: GradualKSModifier
    serialized_modifier = GradualKSModifier.load_obj(
        str(yaml_modifier)
    )  # type: GradualKSModifier
    obj_modifier = GradualKSModifier(
        layers=layers,
        init_sparsity=init_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        param=param,
        inter_func=inter_func,
    )

    assert isinstance(yaml_modifier, GradualKSModifier)
    assert yaml_modifier.layers == serialized_modifier.layers == obj_modifier.layers
    assert (
        yaml_modifier.init_sparsity
        == serialized_modifier.init_sparsity
        == obj_modifier.init_sparsity
    )
    assert (
        yaml_modifier.final_sparsity
        == serialized_modifier.final_sparsity
        == obj_modifier.final_sparsity
    )
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.end_epoch
        == serialized_modifier.end_epoch
        == obj_modifier.end_epoch
    )
    assert (
        yaml_modifier.update_frequency
        == serialized_modifier.update_frequency
        == obj_modifier.update_frequency
    )
    assert yaml_modifier.param == serialized_modifier.param == obj_modifier.param
    assert (
        yaml_modifier.inter_func
        == serialized_modifier.inter_func
        == obj_modifier.inter_func
    )
