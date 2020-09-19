"""
Model function creator classes to be used with estimator
"""

from typing import List, Tuple, Callable, Dict, Any, NamedTuple, Union, Optional
from abc import ABC, abstractmethod
import inspect
from neuralmagicML.tensorflow.recal import (
    NM_RECAL,
    ScheduledModifierManager,
    ConstantKSModifier,
)
from neuralmagicML.tensorflow.utils import tf_compat

__all__ = ["EstimatorModelFn", "ClassificationEstimatorModelFn"]


class MetricUpdateOpsHook(tf_compat.train.SessionRunHook):
    """
    Class to update metric ops for the training mode. Unlike the evaluation
    mode where metrics were specified in eval_metric_ops field of the EstimatorSpec,
    the metric tensors will not be updated automatically for training mode
    tensors, therefore must be done explicitly
    """

    def __init__(self, metrics_update_ops: List[tf_compat.Operation]):
        self._metrics_update_ops = metrics_update_ops

    def after_run(self, run_context, run_values):
        return tf_compat.train.SessionRunArgs(fetches=self._metrics_update_ops)


class RecalOpsUpdateHook(tf_compat.train.SessionRunHook):
    """
    Class to update the recal ops right after the training op runs
    """

    def after_run(self, run_context, run_values):
        try:
            graph = tf_compat.get_default_graph()
            recal_update_op = graph.get_operation_by_name(
                "{}/{}".format(NM_RECAL, ScheduledModifierManager.RECAL_UPDATE)
            )
            return tf_compat.train.SessionRunArgs(fetches=recal_update_op)
        except KeyError:
            pass


class SessionInitializerHook(tf_compat.train.SessionRunHook):
    """
    Class to initialize the session for modifiers
    """

    def __init__(self, manager: ScheduledModifierManager):
        self._manager = manager

    def begin(self):
        if self._manager is not None:
            self._manager.initialize_session()


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

            # Prediction mode
            if mode == tf_compat.estimator.ModeKeys.PREDICT:
                predictions = self.create_predictions(net_outputs, params)
                return tf_compat.estimator.EstimatorSpec(
                    tf_compat.estimator.ModeKeys.PREDICT,
                    predictions=predictions,
                )

            # Train and eval mode
            loss = self.create_loss(net_outputs, labels, params)
            metrics_dict = self.create_metrics(net_outputs, labels, params)
            metric_update_ops_hook = self.create_metric_update_ops_hook(
                metrics_dict, params
            )
            summary_train_hook = self.create_train_summary_hook(metrics_dict, params)

            if mode == tf_compat.estimator.ModeKeys.EVAL:
                return tf_compat.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics_dict
                )
            # Train mode only
            training_op = self.create_training_op(loss, params)
            (
                init_sess_hook,
                mod_update_ops_hook,
            ) = self.create_modifier_ops_and_update_hook(params)

            training_hooks = [metric_update_ops_hook, summary_train_hook]
            if init_sess_hook is not None:
                training_hooks.append(init_sess_hook)
            if mod_update_ops_hook is not None:
                training_hooks.append(mod_update_ops_hook)

            if params["checkpoint_path"] is not None:
                # Finetuning
                base_name_scope = params["base_name_scope"]
                tf_compat.train.init_from_checkpoint(
                    params["checkpoint_path"],
                    {"{}/".format(base_name_scope): "{}/".format(base_name_scope)},
                )

            scaffold = self.create_scaffold(params)
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
    ) -> Dict[str, Tuple[tf_compat.Tensor, tf_compat.Operation]]:
        """
        Create metrics for evaluation

        :param net_outputs: output tensors of the model graph
        :param labels: ground truth labels
        :param params: the model function params
        :return: dictionary of metric tensors and their update operations
        """
        raise NotImplementedError()

    @abstractmethod
    def create_metric_update_ops_hook(self, metrics_dict, params):
        """
        Create hooks for the update operations of the collected metrics

        :param metrics_dict: dictionary of metrics, created as a result of
            create_metrics function
        :param params: the model function params
        :return: a SessionRunHook instance
        """
        raise NotImplementedError()

    @abstractmethod
    def create_train_summary_hook(self, metrics_dict, params):
        """
        Create hook for the summary of metrics

        :param metrics_dict: dictionary of metrics, created as a result of
            create_metrics function
        :param params: the model function params
        :return: a SessionRunHook instance
        """
        raise NotImplementedError()

    @abstractmethod
    def create_training_op(self, loss, params: Dict[str, Any]):
        """
        Create training op for optimization

        :param loss: the loss tensor
        :param params: the model function params
        :return: an Operation minimizing loss
        """
        raise NotImplementedError

    @abstractmethod
    def create_modifier_ops_and_update_hook(self, params: Dict[str, Any]):
        """
        Create modifier ops and their update hook to run

        :param params: the model function params
        :return: a SessionRunHook instance
        """
        raise NotImplementedError()

    @abstractmethod
    def create_scaffold(self, params):
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
    ) -> Dict[str, Tuple[tf_compat.Tensor, tf_compat.Operation]]:
        """
        Create metrics for evaluation

        :param net_outputs: output tensors of the model graph
        :param labels: ground truth labels
        :param params: the model function params
        :return: dictionary of metric tensors and their update operations
        """
        metrics = params.get("metrics", [])

        metrics_dict = {}
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
                else:
                    raise ValueError("Unsupported metric: {}".format(metric))

        return metrics_dict

    def create_metric_update_ops_hook(
        self,
        metrics_dict: Dict[str, Tuple[tf_compat.Tensor, tf_compat.Operation]],
        params: Dict[str, Any],
    ):
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
        metric_update_ops_hook = MetricUpdateOpsHook(metrics_update_ops)
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
            tf_compat.summary.scalar(metric, metrics_dict[metric][1])
        summary_op = tf_compat.summary.merge_all()
        return summary_op

    def create_train_summary_hook(
        self,
        metrics_dict: Dict[str, Tuple[tf_compat.Tensor, tf_compat.Operation]],
        params: Dict[str, Any],
    ):
        """
        Create hook for the summary of metrics

        :param metrics_dict: dictionary of metrics, created as a result of
            create_metrics function
        :param params: the model function params
        :return: a SessionRunHook instance
        """
        summary_op = self.create_summary_op(metrics_dict, params)
        save_steps = params.get("eval_every_n_steps", 100)
        logs_dir = params.get("logs_dir", "logs")
        summary_train_hook = tf_compat.train.SummarySaverHook(
            save_steps=save_steps,
            output_dir=logs_dir,
            summary_op=summary_op,
        )
        return summary_train_hook

    def create_training_op(self, loss: tf_compat.Tensor, params: Dict[str, Any]):
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
            training_op = optimizer.minimize(loss, global_step=global_step)
        return training_op

    def create_modifier_ops_and_update_hook(self, params: Dict[str, Any]):
        """
        Create modifier ops and their update hook to run

        :param params: the model function params
        :return: a SessionRunHook instance
        """
        add_mods = (
            [ConstantKSModifier(params="__ALL__")]
            if params["sparse_transfer_learn"]
            else None
        )

        init_sess_hook = None
        mod_update_ops_hook = None
        recal_config_path = params.get("recal_config_path")
        if recal_config_path is not None:
            global_step = tf_compat.train.get_or_create_global_step()
            steps_per_epoch = params["steps_per_epoch"]
            manager = ScheduledModifierManager.from_yaml(recal_config_path, add_mods)
            init_sess_hook = SessionInitializerHook(manager)
            mod_ops, mod_extras = manager.create_ops(steps_per_epoch, global_step)
            mod_update_ops_hook = RecalOpsUpdateHook() if mod_ops is not None else None
        return init_sess_hook, mod_update_ops_hook

    def create_scaffold(self, params: Dict[str, Any]):
        """
        Create scaffold to be attached to the train estimator spec, containing
        at least the saver

        :param params: the model function params
        :return: a Scaffold instance
        """
        saver = tf_compat.train.Saver()
        scaffold = tf_compat.train.Scaffold(
            saver=saver,
        )
        return scaffold
