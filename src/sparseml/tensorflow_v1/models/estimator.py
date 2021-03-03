# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model function creator classes to be used with estimator
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Union

from sparseml import get_main_logger
from sparseml.tensorflow_v1.optim import (
    ConstantPruningModifier,
    ModifierSessionRunHook,
    ScheduledModifierManager,
)
from sparseml.tensorflow_v1.utils import tf_compat


__all__ = ["EstimatorModelFn", "ClassificationEstimatorModelFn"]

LOGGER = get_main_logger()


class MetricUpdateOpsHook(tf_compat.train.SessionRunHook):
    """
    Class to update metric ops for the training mode. Unlike the evaluation
    mode where metrics were specified in eval_metric_ops field of the EstimatorSpec,
    the metric tensors will not be updated automatically for training mode
    tensors, therefore must be done explicitly
    """

    def __init__(
        self,
        metrics_update_ops: List[tf_compat.Operation],
        metrics_initializer_ops: List[tf_compat.Operation],
        eval_every_n_steps: int,
    ):
        self._metrics_update_ops = metrics_update_ops
        self._metrics_initializer_ops = metrics_initializer_ops
        self._eval_every_n_steps = eval_every_n_steps

    def before_run(self, run_context):
        # Metrics such as accuracy has a dependency link to the dataset
        # iterator through the label tensor, therefore its update must be
        # executed in the same run of the training op; otherwise the iterator
        # would pull an extra batch from the dataset
        return tf_compat.train.SessionRunArgs(fetches=[self._metrics_update_ops])

    def after_run(self, run_context, run_values):
        sess = run_context.session
        global_step = tf_compat.train.get_or_create_global_step()
        global_step_val = sess.run(global_step)
        if global_step_val % self._eval_every_n_steps == 0:
            LOGGER.info(
                "Reset metrics local variables at the end of step {}".format(
                    global_step_val
                )
            )
            sess.run(self._metrics_initializer_ops)


class EstimatorModelFn(ABC):
    """
    Base class for model function creator
    """

    def create(self, model_const: Callable, *args, **kwargs):
        """
        Create a model function to be used to create estimator

        :param model_const: function to create model graph
        :param arg: additional positional arguments passed into model_const
        :param kwargs: additional keyword arguments passed into model_const
        """

        def model_fn(features, labels, mode, params):
            net_outputs = model_const(features, *args, **kwargs)

            ############################
            #
            # Prediction mode
            #
            ############################
            if mode == tf_compat.estimator.ModeKeys.PREDICT:
                predictions = self.create_predictions(net_outputs, params)
                return tf_compat.estimator.EstimatorSpec(
                    tf_compat.estimator.ModeKeys.PREDICT,
                    predictions=predictions,
                )

            ############################
            #
            # Train and eval mode
            #
            ############################

            # Loss function
            loss = self.create_loss(net_outputs, labels, params)

            # Metrics to collect and their initialiers
            metrics_dict, metrics_initializers_dict = self.create_metrics(
                net_outputs, labels, params
            )

            # Modifier ops and extras
            # Note that extras such as for pruning masks are needed for eval mode too
            (
                mod_manager,
                mod_update_ops_hook,
            ) = self.create_modifier_ops_and_update_hook(params)

            if mode == tf_compat.estimator.ModeKeys.EVAL:
                return tf_compat.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics_dict
                )

            # Training op
            training_op = self.create_training_op(loss, params)

            # Ops to update metrics
            # During training, we call these ops to gather metrics on the current batch
            # at the time of evaluation
            metric_update_ops_hook = self.create_metric_update_ops_hook(
                metrics_dict, metrics_initializers_dict, params
            )

            # Summaries to display on Tensorboard
            summary_train_hook = self.create_train_summary_hook(metrics_dict, params)

            training_hooks = [
                mod_update_ops_hook,
                metric_update_ops_hook,
                summary_train_hook,
            ]

            if "checkpoint_path" in params and params["checkpoint_path"] is not None:
                # Finetuning
                base_name_scope = params["base_name_scope"]
                tf_compat.train.init_from_checkpoint(
                    params["checkpoint_path"],
                    {"{}/".format(base_name_scope): "{}/".format(base_name_scope)},
                )

            scaffold = self.create_scaffold(mod_manager, params)
            return tf_compat.estimator.EstimatorSpec(
                mode,
                loss=loss,
                train_op=training_op,
                scaffold=scaffold,
                training_hooks=training_hooks,
            )

        return model_fn

    @abstractmethod
    def create_predictions(
        self,
        net_outputs: Union[tf_compat.Tensor, Dict[str, tf_compat.Tensor]],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create predictions used to define the estimator spec in prediction mode

        :param net_outputs: output tensors of the model graph
        :param params: the model function params. If "apply_softmax" is specified
            in params then softmax is apply to the net outputs
        :return: dictionary of metric tensors
        """
        raise NotImplementedError()

    @abstractmethod
    def create_loss(
        self,
        net_outputs: Union[tf_compat.Tensor, Dict[str, tf_compat.Tensor]],
        labels: Union[tf_compat.Tensor, Dict[str, tf_compat.Tensor]],
        params: Dict[str, Any],
    ) -> tf_compat.Tensor:
        """
        Create loss function

        :param net_outputs: output tensors of the model graph
        :param labels: ground truth labels
        :param params: the model function params
        :return: a loss tensor
        """
        raise NotImplementedError()

    @abstractmethod
    def create_metrics(
        self,
        net_outputs: Union[tf_compat.Tensor, Dict[str, tf_compat.Tensor]],
        labels: Union[tf_compat.Tensor, Dict[str, tf_compat.Tensor]],
        params: Dict[str, Any],
    ) -> (
        Dict[str, Tuple[tf_compat.Tensor, tf_compat.Operation]],
        List[tf_compat.Tensor],
    ):
        """
        Create metrics for evaluation

        :param net_outputs: output tensors of the model graph
        :param labels: ground truth labels
        :param params: the model function params
        :return: (1) dictionary of metric tensors and their update operations;
                 (2) list of extra/internal vars created for the metrics if any
        """
        raise NotImplementedError()

    @abstractmethod
    def create_metric_update_ops_hook(
        self,
        metrics_dict: Dict[str, Tuple[tf_compat.Tensor, tf_compat.Operation]],
        metrics_initializers_dict: Dict[str, List[tf_compat.Tensor]],
        params: Dict[str, Any],
    ) -> MetricUpdateOpsHook:
        """
        Create hooks for the update operations of the collected metrics

        :param metrics_dict: dictionary of metrics, created as a result of
            create_metrics function
        :param params: the model function params
        :return: a SessionRunHook instance
        """
        raise NotImplementedError()

    @abstractmethod
    def create_train_summary_hook(
        self,
        metrics_dict: Dict[str, Tuple[tf_compat.Tensor, tf_compat.Operation]],
        params: Dict[str, Any],
    ) -> tf_compat.train.SummarySaverHook:
        """
        Create hook for the summary of metrics

        :param metrics_dict: dictionary of metrics, created as a result of
            create_metrics function
        :param params: the model function params
        :return: a SummarySaverHook instance
        """
        raise NotImplementedError()

    @abstractmethod
    def create_training_op(
        self, loss: tf_compat.Tensor, params: Dict[str, Any]
    ) -> tf_compat.Operation:
        """
        Create training op for optimization

        :param loss: the loss tensor
        :param params: the model function params
        :return: an Operation minimizing loss
        """
        raise NotImplementedError

    @abstractmethod
    def create_modifier_ops_and_update_hook(
        self, params: Dict[str, Any]
    ) -> (ScheduledModifierManager, ModifierSessionRunHook):
        """
        Create modifier ops and their update hook to run

        :param params: the model function params
        :return: a SessionRunHook instance
        """
        raise NotImplementedError()

    @abstractmethod
    def create_scaffold(
        self, modifier_manager: ScheduledModifierManager, params: Dict[str, Any]
    ) -> tf_compat.train.Scaffold:
        """
        Create scaffold to be attached to the train estimator spec, containing
        at least the saver

        :param params: the model function params
        :return: a Scaffold instance
        """
        raise NotImplementedError()


class ClassificationEstimatorModelFn(EstimatorModelFn):
    """
    Model function creator for classification models
    """

    def create_predictions(
        self,
        net_outputs: Union[tf_compat.Tensor, Dict[str, tf_compat.Tensor]],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create predictions used to define the estimator spec in prediction mode

        :param net_outputs: output tensors of the model graph
        :param params: the model function params
        :return: dictionary of metric tensors
        """
        apply_softmax = params.get("apply_softmax")
        class_type = params.get("class_type")
        if apply_softmax is not None and class_type is None:
            probabilities = tf_compat.nn.softmax(net_outputs)
        else:
            probabilities = net_outputs
        predictions = {
            "probabilities": probabilities,
            "net_outputs": net_outputs,
        }
        return predictions

    def create_loss(
        self,
        net_outputs: Union[tf_compat.Tensor, Dict[str, tf_compat.Tensor]],
        labels: Union[tf_compat.Tensor, Dict[str, tf_compat.Tensor]],
        params: Dict[str, Any],
    ) -> tf_compat.Tensor:
        """
        Create loss function

        :param net_outputs: output tensors of the model graph
        :param labels: ground truth labels
        :param params: the model function params
        :return: a loss tensor
        """
        loss = params.get("loss")
        with tf_compat.name_scope("loss"):
            if loss == "cross_entropy":
                xentropy = tf_compat.nn.softmax_cross_entropy_with_logits_v2(
                    labels=labels, logits=net_outputs
                )
                loss_tens = tf_compat.reduce_mean(xentropy, name="loss")
            else:
                raise ValueError("Unsupported loss function: {}".format(loss))
        return loss_tens

    def create_metrics(
        self,
        net_outputs: Union[tf_compat.Tensor, Dict[str, tf_compat.Tensor]],
        labels: Union[tf_compat.Tensor, Dict[str, tf_compat.Tensor]],
        params: Dict[str, Any],
    ) -> (
        Dict[str, Tuple[tf_compat.Tensor, tf_compat.Operation]],
        Dict[str, tf_compat.Operation],
    ):
        """
        Create metrics for evaluation

        :param net_outputs: output tensors of the model graph
        :param labels: ground truth labels
        :param params: the model function params
        :return: dictionary of metrics and their reset operations
        """
        metrics = params.get("metrics", [])

        metrics_dict = {}
        metrics_initializers_dict = {}
        with tf_compat.name_scope("metrics"):
            for metric in metrics:
                if metric == "accuracy":
                    labels_argmax = tf_compat.argmax(labels, 1)
                    net_outputs_argmax = tf_compat.argmax(net_outputs, 1)
                    metrics_dict["accuracy"] = tf_compat.metrics.accuracy(
                        labels_argmax,
                        net_outputs_argmax,
                        name="accuracy_metric",
                    )
                    # The total and count variables created to support accuracy
                    running_vars = tf_compat.get_collection(
                        tf_compat.GraphKeys.LOCAL_VARIABLES,
                        scope="metrics/accuracy_metric",
                    )
                    running_vars_initializer = tf_compat.variables_initializer(
                        var_list=running_vars
                    )
                    metrics_initializers_dict[metric] = running_vars_initializer
                else:
                    raise ValueError("Unsupported metric: {}".format(metric))

        return (metrics_dict, metrics_initializers_dict)

    def create_metric_update_ops_hook(
        self,
        metrics_dict: Dict[str, Tuple[tf_compat.Tensor, tf_compat.Operation]],
        metrics_initializers_dict: Dict[str, List[tf_compat.Tensor]],
        params: Dict[str, Any],
    ) -> MetricUpdateOpsHook:
        """
        Create hooks for the update operations of the collected metrics

        :param metrics_dict: dictionary of metrics, created as a result of
            create_metrics function
        :param params: the model function params
        :return: a SessionRunHook instance
        """
        metrics_update_ops = [
            metrics_dict[metric_key][1] for metric_key in metrics_dict
        ]
        metrics_initializer_ops = [
            metrics_initializers_dict[metric_key]
            for metric_key in metrics_initializers_dict
        ]
        metric_update_ops_hook = MetricUpdateOpsHook(
            metrics_update_ops, metrics_initializer_ops, params["eval_every_n_steps"]
        )
        return metric_update_ops_hook

    def create_summary_op(
        self,
        metrics_dict: Dict[str, Tuple[tf_compat.Tensor, tf_compat.Operation]],
        params: Dict[str, Any],
    ):
        """
        Create summary op given metric dictionary
        """
        for metric in metrics_dict:
            metric_tensor = metrics_dict[metric][0]
            tf_compat.summary.scalar(metric, metric_tensor)
        summary_op = tf_compat.summary.merge_all()
        return summary_op

    def create_train_summary_hook(
        self,
        metrics_dict: Dict[str, Tuple[tf_compat.Tensor, tf_compat.Operation]],
        params: Dict[str, Any],
    ) -> tf_compat.train.SummarySaverHook:
        """
        Create hook for the summary of metrics

        :param metrics_dict: dictionary of metrics, created as a result of
            create_metrics function
        :param params: the model function params
        :return: a SsummarySaverHook instance
        """
        summary_op = self.create_summary_op(metrics_dict, params)
        save_steps = params.get("eval_every_n_steps", params["eval_every_n_steps"])
        logs_dir = params.get("logs_dir", "logs")
        summary_train_hook = tf_compat.train.SummarySaverHook(
            save_steps=save_steps,
            output_dir=logs_dir,
            summary_op=summary_op,
        )
        return summary_train_hook

    def create_training_op(
        self, loss: tf_compat.Tensor, params: Dict[str, Any]
    ) -> tf_compat.Operation:
        """
        Create training op for optimization

        :param loss: the loss tensor
        :param params: the model function params
        :return: an Operation minimizing loss
        """
        global_step = tf_compat.train.get_or_create_global_step()

        optimizer_const = {}
        for opt_name in dir(tf_compat.train):
            opt_cls = getattr(tf_compat.train, opt_name)
            if inspect.isclass(opt_cls) and issubclass(
                opt_cls, tf_compat.train.Optimizer
            ):
                optimizer_const[opt_name] = opt_cls

        optimizer_name = params.get("optimizer", "AdamOptimizer")
        if optimizer_name not in optimizer_const:
            raise ValueError("Unsupported optimizer: {}".format(optimizer_name))
        optimizer_params = params.get("optimizer_params", {})
        optimizer = optimizer_const[optimizer_name](**optimizer_params)

        with tf_compat.name_scope("train"):
            # We are using tf.layers.batch_normalization to support previous versions
            # of TF, which requires us explicite model the dependency between the
            # update of moving average and variance with training op
            update_ops = tf_compat.get_collection(tf_compat.GraphKeys.UPDATE_OPS)
            with tf_compat.control_dependencies(update_ops):
                training_op = optimizer.minimize(loss, global_step=global_step)
        return training_op

    def create_modifier_ops_and_update_hook(
        self, params: Dict[str, Any]
    ) -> (ScheduledModifierManager, ModifierSessionRunHook):
        """
        Create modifier ops and their update hook to run

        :param params: the model function params
        :return: a SessionRunHook instance
        """
        add_mods = (
            [ConstantPruningModifier(params="__ALL__")]
            if "sparse_transfer_learn" in params and params["sparse_transfer_learn"]
            else None
        )

        mod_update_ops_hook = None
        manager = None
        recipe_path = params.get("recipe_path")
        if recipe_path is not None:
            global_step = tf_compat.train.get_or_create_global_step()
            steps_per_epoch = params["steps_per_epoch"]
            manager = ScheduledModifierManager.from_yaml(recipe_path, add_mods)
            mod_ops, mod_extras = manager.create_ops(steps_per_epoch, global_step)
            mod_update_ops_hook = ModifierSessionRunHook(mod_ops)
        return manager, mod_update_ops_hook

    def create_scaffold(
        self, modifier_manager: ScheduledModifierManager, params: Dict[str, Any]
    ) -> tf_compat.train.Scaffold:
        """
        Create scaffold to be attached to the train estimator spec, containing
        at least the saver

        :param params: the model function params
        :return: a Scaffold instance
        """

        def init_fn(scaffold, session):
            if modifier_manager is None:
                return
            for mod in modifier_manager.modifiers:
                mod.initialize_session(session)

        saver = tf_compat.train.Saver()
        scaffold = tf_compat.train.Scaffold(saver=saver, init_fn=init_fn)
        return scaffold
