from typing import Callable
import pytest
import numpy as np

from neuralmagicML.tensorflow.utils import tf_compat

from neuralmagicML.tensorflow.recal.modifier import (
    EXTRAS_KEY_LEARNING_RATE,
    EXTRAS_KEY_SUMMARIES,
)

from neuralmagicML.tensorflow.recal import (
    ScheduledModifierManager,
    LearningRateModifier,
    SetLearningRateModifier,
)

from tests.tensorflow.recal.test_modifier import (
    ScheduledModifierTest,
    mlp_graph_lambda,
)


EPSILON = 1e-7

adam = tf_compat.train.AdamOptimizer
adagrad = tf_compat.train.AdagradOptimizer


@pytest.mark.parametrize(
    "graph_lambda,modifier_lambda",
    [
        (
            mlp_graph_lambda,
            lambda: SetLearningRateModifier(learning_rate=0.1, start_epoch=0),
        ),
        (
            mlp_graph_lambda,
            lambda: SetLearningRateModifier(learning_rate=0.5, start_epoch=0),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestSetLRModifiers_NoManager(ScheduledModifierTest):
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], SetLearningRateModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()
        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()

            # Further set up for loss, optimizer and training op
            X_placeholder = graph.get_tensor_by_name("inp:0")
            y_hat = graph.get_tensor_by_name("out:0")
            n_inputs = X_placeholder.shape[1]
            n_outputs = y_hat.shape[1]
            y_placeholder = tf_compat.placeholder(
                tf_compat.float32, shape=(None, n_outputs), name="y"
            )
            ops, extras = modifier.create_ops(
                steps_per_epoch, global_step=global_step, graph=graph
            )
            assert len(ops) == 0
            assert len(extras) == 2
            assert EXTRAS_KEY_LEARNING_RATE in extras
            assert EXTRAS_KEY_SUMMARIES in extras

            learning_rate = extras["learning_rate"]
            with tf_compat.name_scope("train"):
                optimizer = adam(learning_rate=learning_rate)
                loss = tf_compat.losses.mean_squared_error(y_placeholder, y_hat)
                training_op = optimizer.minimize(loss, global_step=global_step)

        np.random.seed(12)
        n_samples = 1000
        batch_size = 8
        epochs = 5
        X = np.random.rand(n_samples, n_inputs)
        y = np.random.rand(n_samples, n_outputs)

        batches = int(n_samples / batch_size)
        with tf_compat.Session(graph=graph) as sess:
            sess.run(tf_compat.global_variables_initializer())
            for epoch in range(epochs):
                for batch in range(batches):
                    indices = np.random.choice(range(n_samples), batch_size)
                    X_batch, y_batch = X[indices], y[indices]
                    gs = sess.run(global_step)
                    expected = modifier.learning_rate
                    optim_lr = sess.run(_get_lr(optimizer))
                    assert abs(optim_lr - expected) <= EPSILON, "Failed at {}".format(
                        gs
                    )
                    sess.run(
                        training_op,
                        feed_dict={X_placeholder: X_batch, y_placeholder: y_batch},
                    )


@pytest.mark.parametrize(
    "graph_lambda,modifier_lambda",
    [
        (
            mlp_graph_lambda,
            lambda: LearningRateModifier(
                lr_class="ExponentialLR",
                lr_kwargs={"gamma": 0.9},
                start_epoch=0,
                end_epoch=3,
                init_lr=0.1,
            ),
        ),
        (
            mlp_graph_lambda,
            lambda: LearningRateModifier(
                lr_class="ExponentialLR",
                lr_kwargs={"gamma": 0.5},
                start_epoch=0,
                end_epoch=5.5,
                init_lr=0.1,
            ),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestExponentialLRModifiers_NoManager(ScheduledModifierTest):
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], SetLearningRateModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()
        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()

            # Further set up for loss, optimizer and training op
            X_placeholder = graph.get_tensor_by_name("inp:0")
            y_hat = graph.get_tensor_by_name("out:0")
            n_inputs = X_placeholder.shape[1]
            n_outputs = y_hat.shape[1]
            y_placeholder = tf_compat.placeholder(
                tf_compat.float32, shape=(None, n_outputs), name="y"
            )
            ops, extras = modifier.create_ops(
                steps_per_epoch, global_step=global_step, graph=graph
            )
            assert len(ops) == 0
            assert len(extras) == 2
            assert EXTRAS_KEY_LEARNING_RATE in extras
            assert EXTRAS_KEY_SUMMARIES in extras

            learning_rate = extras["learning_rate"]
            with tf_compat.name_scope("train"):
                optimizer = adam(learning_rate=learning_rate)
                loss = tf_compat.losses.mean_squared_error(y_placeholder, y_hat)
                training_op = optimizer.minimize(loss, global_step=global_step)

        np.random.seed(12)
        n_samples = 1000
        batch_size = 8
        epochs = 10
        X = np.random.rand(n_samples, n_inputs)
        y = np.random.rand(n_samples, n_outputs)

        end_step = modifier.end_epoch * steps_per_epoch
        init_lr = modifier.init_lr
        gamma = modifier.lr_kwargs["gamma"]
        batches = int(n_samples / batch_size)
        with tf_compat.Session(graph=graph) as sess:
            sess.run(tf_compat.global_variables_initializer())
            for epoch in range(epochs):
                for batch in range(batches):
                    indices = np.random.choice(range(n_samples), batch_size)
                    X_batch, y_batch = X[indices], y[indices]
                    gs = sess.run(global_step)
                    if gs == 0:
                        expected = init_lr
                    elif gs <= end_step:
                        expected = expected * gamma
                    optim_lr = sess.run(_get_lr(optimizer))
                    assert abs(optim_lr - expected) <= EPSILON, "Failed at {}".format(
                        gs
                    )
                    sess.run(
                        training_op,
                        feed_dict={X_placeholder: X_batch, y_placeholder: y_batch},
                    )


@pytest.mark.parametrize(
    "graph_lambda,modifier_lambda",
    [
        (
            mlp_graph_lambda,
            lambda: LearningRateModifier(
                lr_class="StepLR",
                lr_kwargs={"gamma": 0.9, "step": 10},
                start_epoch=0,
                end_epoch=3,
                init_lr=0.1,
            ),
        ),
        (
            mlp_graph_lambda,
            lambda: LearningRateModifier(
                lr_class="StepLR",
                lr_kwargs={"gamma": 0.5, "step": 15},
                start_epoch=0,
                end_epoch=5.5,
                init_lr=0.2,
            ),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestStepLRModifiers_NoManager(ScheduledModifierTest):
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], SetLearningRateModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()
        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()

            # Further set up for loss, optimizer and training op
            X_placeholder = graph.get_tensor_by_name("inp:0")
            y_hat = graph.get_tensor_by_name("out:0")
            n_inputs = X_placeholder.shape[1]
            n_outputs = y_hat.shape[1]
            y_placeholder = tf_compat.placeholder(
                tf_compat.float32, shape=(None, n_outputs), name="y"
            )
            ops, extras = modifier.create_ops(
                steps_per_epoch, global_step=global_step, graph=graph
            )
            assert len(ops) == 0
            assert len(extras) == 2
            assert EXTRAS_KEY_LEARNING_RATE in extras
            assert EXTRAS_KEY_SUMMARIES in extras

            learning_rate = extras["learning_rate"]
            with tf_compat.name_scope("train"):
                optimizer = adam(learning_rate=learning_rate)
                loss = tf_compat.losses.mean_squared_error(y_placeholder, y_hat)
                training_op = optimizer.minimize(loss, global_step=global_step)

        np.random.seed(12)
        n_samples = 1000
        batch_size = 8
        epochs = 5
        X = np.random.rand(n_samples, n_inputs)
        y = np.random.rand(n_samples, n_outputs)

        end_step = modifier.end_epoch * steps_per_epoch
        init_lr = modifier.init_lr
        gamma = modifier.lr_kwargs["gamma"]
        decay_steps = modifier.lr_kwargs["step"]
        batches = int(n_samples / batch_size)
        with tf_compat.Session(graph=graph) as sess:
            sess.run(tf_compat.global_variables_initializer())
            for epoch in range(epochs):
                for batch in range(batches):
                    indices = np.random.choice(range(n_samples), batch_size)
                    X_batch, y_batch = X[indices], y[indices]
                    gs = sess.run(global_step)
                    if gs == 0:
                        expected = init_lr
                    elif gs <= end_step:
                        expected = init_lr * (gamma ** int(gs / decay_steps))
                    optim_lr = sess.run(_get_lr(optimizer))
                    assert abs(optim_lr - expected) <= EPSILON, "Failed at {}".format(
                        gs
                    )
                    sess.run(
                        training_op,
                        feed_dict={X_placeholder: X_batch, y_placeholder: y_batch},
                    )


@pytest.mark.parametrize(
    "graph_lambda,modifier_lambda",
    [
        (
            mlp_graph_lambda,
            lambda: LearningRateModifier(
                lr_class="MultiStepLR",
                lr_kwargs={"gamma": 0.9, "milestones": [1, 2.5]},
                start_epoch=0,
                end_epoch=3,
                init_lr=0.1,
            ),
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("steps_per_epoch", [100], scope="function")
class TestMultiStepLRModifiers_NoManager(ScheduledModifierTest):
    def test_lifecycle(
        self,
        modifier_lambda: Callable[[], LearningRateModifier],
        graph_lambda: Callable[[], tf_compat.Graph],
        steps_per_epoch: int,
    ):
        modifier = modifier_lambda()
        graph = graph_lambda()
        with graph.as_default():
            global_step = tf_compat.train.get_or_create_global_step()

            # Further set up for loss, optimizer and training op
            X_placeholder = graph.get_tensor_by_name("inp:0")
            y_hat = graph.get_tensor_by_name("out:0")
            n_inputs = X_placeholder.shape[1]
            n_outputs = y_hat.shape[1]
            y_placeholder = tf_compat.placeholder(
                tf_compat.float32, shape=(None, n_outputs), name="y"
            )
            ops, extras = modifier.create_ops(
                steps_per_epoch, global_step=global_step, graph=graph
            )
            assert len(ops) == 0
            assert len(extras) == 2
            assert EXTRAS_KEY_LEARNING_RATE in extras
            assert EXTRAS_KEY_SUMMARIES in extras

            learning_rate = extras["learning_rate"]
            with tf_compat.name_scope("train"):
                optimizer = adam(learning_rate=learning_rate)
                loss = tf_compat.losses.mean_squared_error(y_placeholder, y_hat)
                training_op = optimizer.minimize(loss, global_step=global_step)

        np.random.seed(12)
        n_samples = 100
        batch_size = 4
        epochs = 5
        X = np.random.rand(n_samples, n_inputs)
        y = np.random.rand(n_samples, n_outputs)

        gamma = modifier.lr_kwargs["gamma"]
        boundaries = [
            round(steps_per_epoch * v) for v in modifier.lr_kwargs["milestones"]
        ]
        values = [modifier.init_lr * (gamma ** i) for i in range(len(boundaries) + 1)]

        batches = int(n_samples / batch_size)

        with tf_compat.Session(graph=graph) as sess:
            sess.run(tf_compat.global_variables_initializer())
            for epoch in range(epochs):
                for batch in range(batches):
                    indices = np.random.choice(range(n_samples), batch_size)
                    X_batch, y_batch = X[indices], y[indices]
                    gs = sess.run(global_step)
                    delta_gs = gs - modifier.start_epoch * steps_per_epoch
                    expected = None
                    for i in range(len(values) - 1):
                        if delta_gs <= boundaries[i]:
                            expected = values[i]
                            break
                    expected = values[-1] if expected is None else expected
                    optim_lr = sess.run(_get_lr(optimizer))
                    assert abs(optim_lr - expected) <= EPSILON, "Failed at {}".format(
                        gs
                    )
                    sess.run(
                        training_op,
                        feed_dict={X_placeholder: X_batch, y_placeholder: y_batch},
                    )


###########################################################################################
#
# Test using modifiers through the manager
#
###########################################################################################


def simple_model(n_inputs: int, n_neurons: int):
    """
    Model used for all the tests
    """
    tf_compat.reset_default_graph()
    X = tf_compat.placeholder(tf_compat.float32, shape=(None, n_inputs), name="X")
    y = tf_compat.placeholder(tf_compat.float32, shape=(None, n_neurons), name="y")
    W = tf_compat.Variable(
        np.ones([n_inputs, n_neurons]),
        trainable=True,
        name="kernel",
        dtype=tf_compat.float32,
    )
    b = tf_compat.Variable(
        tf_compat.zeros([n_neurons]),
        trainable=True,
        name="bias",
        dtype=tf_compat.float32,
    )
    y_hat = tf_compat.matmul(X, W) + b
    loss = tf_compat.losses.mean_squared_error(y, y_hat)
    return tf_compat.get_default_graph(), loss, X, y


# Parameters defining built-in schedulers used for testing

exp_lr_spec = {"lr_class": "ExponentialLR", "lr_kwargs": {"gamma": 0.9}, "init_lr": 0.1}

step_lr_spec = {
    "lr_class": "StepLR",
    "lr_kwargs": {"gamma": 0.9, "step": 10},
    "init_lr": 0.2,
}

multistep_lr_spec = {
    "lr_class": "MultiStepLR",
    "lr_kwargs": {"gamma": 0.9, "milestones": [2, 3.5]},
    "init_lr": 0.25,
}

setlr_01_spec = {"learning_rate": 0.1}

setlr_02_spec = {"learning_rate": 0.5}


# Reimplementation of the above schedulers, served as ground
# truth for testing


def _exp_lr(step):
    """
    ExponentialDecay
    """
    return exp_lr_spec["init_lr"] * (exp_lr_spec["lr_kwargs"]["gamma"] ** step)


def _step_lr(step):
    """
    StepLR
    """
    gamma = step_lr_spec["lr_kwargs"]["gamma"]
    decay_steps = step_lr_spec["lr_kwargs"]["step"]
    return step_lr_spec["init_lr"] * (gamma ** int(step / decay_steps))


def _multistep_lr(step, steps_per_epoch):
    """
    StepLR
    """
    init_lr = multistep_lr_spec["init_lr"]
    gamma = multistep_lr_spec["lr_kwargs"]["gamma"]
    milestones = multistep_lr_spec["lr_kwargs"]["milestones"]
    boundaries = [round(v * steps_per_epoch) for v in milestones]
    values = [init_lr * (gamma ** i) for i in range(len(boundaries) + 1)]

    expected = None
    for i in range(len(values) - 1):
        if step <= boundaries[i]:
            expected = values[i]
            break
    expected = values[-1] if expected is None else expected
    return expected


def _setlr_01():
    """
    SetLearningRateModifier
    """
    return setlr_01_spec["learning_rate"]


def _setlr_02():
    """
    SetLearningRateModifier
    """
    return setlr_02_spec["learning_rate"]


def expected_lr(current_step, start_steps, end_steps, steps_per_epoch, scheduler_funcs):
    """
    Get expected learning rate at a global step

    :param current_step: The current global step
    :param start_end_steps: List of start, end steps of the sequence of modifiers
    :param scheduler_funcs: Functions defined behaviors of schedulers
    :return A value served as expected learning rate at the current step
    """
    assert len(scheduler_funcs) == len(start_steps) == len(end_steps)
    idx = 0
    while idx < len(start_steps) - 1 and not (
        start_steps[idx] <= current_step < start_steps[idx + 1]
    ):
        idx += 1
    if current_step < end_steps[idx]:
        step = current_step - start_steps[idx]
    else:
        step = end_steps[idx] - start_steps[idx]
    fn = scheduler_funcs[idx]
    if fn == _exp_lr or fn == _step_lr:
        return fn(step)
    elif fn == _multistep_lr:
        return fn(step, steps_per_epoch)
    else:
        return fn()


@pytest.mark.parametrize(
    "modifier_specs, scheduler_funcs",
    [
        ([(setlr_01_spec, 0, -1, "SetLearningRateModifier")], [_setlr_01]),
        ([(setlr_02_spec, 0, -1, "SetLearningRateModifier")], [_setlr_02]),
        ([(exp_lr_spec, 0, 3, "LearningRateModifier")], [_exp_lr]),
        ([(step_lr_spec, 0, 2.5, "LearningRateModifier")], [_step_lr]),
        (
            [
                (exp_lr_spec, 0, 2.5, "LearningRateModifier"),
                (step_lr_spec, 2.5, 4, "LearningRateModifier"),
            ],
            [_exp_lr, _step_lr],
        ),
        (
            [
                (exp_lr_spec, 0, 2.5, "LearningRateModifier"),
                (step_lr_spec, 3.5, 4, "LearningRateModifier"),
            ],
            [_exp_lr, _step_lr],
        ),
        (
            [
                (exp_lr_spec, 0, 2.5, "LearningRateModifier"),
                (step_lr_spec, 3.5, 4, "LearningRateModifier"),
                (multistep_lr_spec, 3.75, 5, "LearningRateModifier"),
            ],
            [_exp_lr, _step_lr, _multistep_lr],
        ),
        (
            [
                (setlr_01_spec, 0, -1, "SetLearningRateModifier"),
                (setlr_02_spec, 3, -1, "SetLearningRateModifier"),
            ],
            [_setlr_01, _setlr_02],
        ),
        (
            [
                (setlr_01_spec, 0, -1, "SetLearningRateModifier"),
                (setlr_02_spec, 1, -1, "SetLearningRateModifier"),
                (exp_lr_spec, 2.5, 3.75, "LearningRateModifier"),
                (multistep_lr_spec, 4, 4.5, "LearningRateModifier"),
            ],
            [_setlr_01, _setlr_02, _exp_lr, _multistep_lr],
        ),
    ],
)
@pytest.mark.parametrize(
    "n_samples, batch_size, epochs, steps_per_epoch",
    [(1000, 8, 5, 100), (1000, 16, 5, 57), (1555, 11, 5, 100)],
)
@pytest.mark.parametrize("optim_cls", [adam, adagrad])
def test_lr_modifiers_with_manager(
    modifier_specs,
    scheduler_funcs,
    n_samples,
    batch_size,
    epochs,
    steps_per_epoch,
    optim_cls,
):
    modifiers = []
    start_steps = []
    end_steps = []

    for (mod_spec, start_epoch, end_epoch, mod_cls) in modifier_specs:
        if mod_cls == "LearningRateModifier":
            mod = LearningRateModifier(
                mod_spec["lr_class"],
                mod_spec["lr_kwargs"],
                init_lr=mod_spec["init_lr"],
                start_epoch=start_epoch,
                end_epoch=end_epoch,
            )
            assert mod.start_epoch < mod.end_epoch <= epochs
        elif mod_cls == "SetLearningRateModifier":
            mod = SetLearningRateModifier(
                mod_spec["learning_rate"], start_epoch=start_epoch, end_epoch=end_epoch
            )
            assert mod.start_epoch < epochs
        modifiers.append(mod)
        start_steps.append(round(start_epoch * steps_per_epoch))
        end_steps.append(round(end_epoch * steps_per_epoch))

    n_inputs = 10
    n_outputs = 1
    graph, loss, X_placeholder, y_placeholder = simple_model(n_inputs, n_outputs)

    manager = ScheduledModifierManager(modifiers)
    ops, extras = manager.create_ops(steps_per_epoch, graph=graph)
    learning_rate = extras["learning_rate"]

    global_step = tf_compat.train.get_or_create_global_step()

    with tf_compat.name_scope("train"):
        optimizer = optim_cls(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss, global_step=global_step)

    np.random.seed(12)
    X = np.random.rand(n_samples, n_inputs)
    y = np.random.rand(n_samples, n_outputs)

    batches = int(n_samples / batch_size)
    with tf_compat.Session() as sess:
        sess.run(tf_compat.global_variables_initializer())
        for epoch in range(epochs):
            for batch in range(batches):
                indices = np.random.choice(range(n_samples), batch_size)
                X_batch, y_batch = X[indices], y[indices]
                gs = sess.run(global_step)
                expected = expected_lr(
                    gs, start_steps, end_steps, steps_per_epoch, scheduler_funcs
                )
                optim_lr = sess.run(_get_lr(optimizer))
                assert abs(optim_lr - expected) <= EPSILON, "Failed at {}".format(gs)
                sess.run(
                    training_op,
                    feed_dict={X_placeholder: X_batch, y_placeholder: y_batch},
                )


def test_set_lr_yaml():
    start_epoch = 10.0
    set_lr = 0.01
    yaml_str = """
    !SetLearningRateModifier
        learning_rate: {}
        start_epoch: {}
    """.format(
        set_lr, start_epoch
    )
    yaml_modifier = SetLearningRateModifier.load_obj(
        yaml_str
    )  # type: SetLearningRateModifier
    serialized_modifier = SetLearningRateModifier.load_obj(
        str(yaml_modifier)
    )  # type: SetLearningRateModifier
    obj_modifier = SetLearningRateModifier(learning_rate=0.01, start_epoch=start_epoch)

    assert isinstance(yaml_modifier, SetLearningRateModifier)
    assert (
        yaml_modifier.learning_rate
        == serialized_modifier.learning_rate
        == obj_modifier.learning_rate
    )
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )


def test_expo_lr_yaml():
    yaml_str = """
        !LearningRateModifier
            lr_class: ExponentialLR
            lr_kwargs:
                gamma: 0.96
            init_lr: 0.1
            start_epoch: 0.0
            end_epoch: 10.0
            log_types: __ALL__
    """
    yaml_modifier = LearningRateModifier.load_obj(
        yaml_str
    )  # type: LearningRateModifier
    serialized_modifier = LearningRateModifier.load_obj(
        str(yaml_modifier)
    )  # type: LearningRateModifier
    obj_modifier = LearningRateModifier(
        lr_class="ExponentialLR",
        lr_kwargs={"gamma": 0.96},
        init_lr=0.1,
        start_epoch=0,
        end_epoch=10,
    )
    assert isinstance(yaml_modifier, LearningRateModifier)
    assert (
        yaml_modifier._lr_class
        == serialized_modifier._lr_class
        == obj_modifier._lr_class
    )
    assert (
        yaml_modifier._lr_kwargs
        == serialized_modifier._lr_kwargs
        == obj_modifier._lr_kwargs
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

    assert yaml_modifier.init_lr == serialized_modifier.init_lr == obj_modifier.init_lr


def test_step_lr_yaml():
    yaml_str = """
        !LearningRateModifier
            lr_class: StepLR
            lr_kwargs:
                gamma: 0.96
                step: 10
            init_lr: 0.1
            start_epoch: 0.0
            end_epoch: 10.0
            log_types: __ALL__
    """
    yaml_modifier = LearningRateModifier.load_obj(
        yaml_str
    )  # type: LearningRateModifier
    serialized_modifier = LearningRateModifier.load_obj(
        str(yaml_modifier)
    )  # type: LearningRateModifier
    obj_modifier = LearningRateModifier(
        lr_class="StepLR",
        lr_kwargs={"gamma": 0.96, "step": 10},
        init_lr=0.1,
        start_epoch=0,
        end_epoch=10,
    )
    assert isinstance(yaml_modifier, LearningRateModifier)
    assert (
        yaml_modifier._lr_class
        == serialized_modifier._lr_class
        == obj_modifier._lr_class
    )
    assert (
        yaml_modifier._lr_kwargs
        == serialized_modifier._lr_kwargs
        == obj_modifier._lr_kwargs
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

    assert yaml_modifier.init_lr == serialized_modifier.init_lr == obj_modifier.init_lr


def test_multi_step_lr_yaml():
    yaml_str = """
        !LearningRateModifier
            lr_class: MultiStepLR
            lr_kwargs:
                gamma: 0.96
                milestones: [2, 3.5]
            init_lr: 0.1
            start_epoch: 0.0
            end_epoch: 10.0
            log_types: __ALL__
    """
    yaml_modifier = LearningRateModifier.load_obj(
        yaml_str
    )  # type: LearningRateModifier
    serialized_modifier = LearningRateModifier.load_obj(
        str(yaml_modifier)
    )  # type: LearningRateModifier
    obj_modifier = LearningRateModifier(
        lr_class="MultiStepLR",
        lr_kwargs={"gamma": 0.96, "milestones": [2, 3.5]},
        init_lr=0.1,
        start_epoch=0,
        end_epoch=10,
    )
    assert isinstance(yaml_modifier, LearningRateModifier)
    assert (
        yaml_modifier._lr_class
        == serialized_modifier._lr_class
        == obj_modifier._lr_class
    )
    assert (
        yaml_modifier._lr_kwargs
        == serialized_modifier._lr_kwargs
        == obj_modifier._lr_kwargs
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

    assert yaml_modifier.init_lr == serialized_modifier.init_lr == obj_modifier.init_lr


def _get_lr(optim) -> tf_compat.Variable:
    if hasattr(optim, "_learning_rate"):
        return optim._learning_rate
    if hasattr(optim, "_lr"):
        return optim._lr
    raise ValueError("Internal LR not found")
