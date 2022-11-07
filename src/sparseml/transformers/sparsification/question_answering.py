# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/huggingface/transformers
# neuralmagic: no copyright

"""
Training and post-processing utilities for question answering.
"""

import collections
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.nn import Module
from tqdm.auto import tqdm
from transformers import is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput

from sparseml.transformers.sparsification.trainer import (
    TrainerInterface,
    TransformersTrainer,
)


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


__all__ = [
    "QuestionAnsweringTrainer",
    "postprocess_qa_predictions",
]


_LOGGER = logging.getLogger(__name__)


class _QuestionAnsweringTrainer(TransformersTrainer):
    """
    Trainer implementation for Question-Answering processing
    """

    def __init__(
        self,
        *args,
        eval_examples=None,
        post_process_function=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(
        self,
        eval_dataset=None,
        eval_examples=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Always evaluate w/ fp32 to be closer to DeepSparse
        use_cuda_amp = self.use_cuda_amp
        if not self.args.fp16_full_eval and not self.args.bf16_full_eval:
            self.use_cuda_amp = False

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )

        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics,
                # otherwise we defer to self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions
            )
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA
            # (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        self.use_cuda_amp = use_cuda_amp

        return metrics

    def predict(
        self,
        predict_dataset,
        predict_examples,
        ignore_keys=None,
        metric_key_prefix: str = "test",
    ):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )

        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics,
                # otherwise we defer to self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(
            predict_examples, predict_dataset, output.predictions, "predict"
        )
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(
            predictions=predictions.predictions,
            label_ids=predictions.label_ids,
            metrics=metrics,
        )


class QuestionAnsweringTrainer(TrainerInterface, _QuestionAnsweringTrainer):
    """
    Trainer for running sparsification recipes with Question Answering training

    :param model: the model to use with the trainer and apply sparsification to
    :param model_state_path: the state path to the model,
        used to load config and tokenizer settings
    :param recipe: the recipe, if any, to apply to the modle and training
        process
    :param recipe_args: A json string, csv key=value string, or dictionary containing
        arguments to override the root arguments within the recipe such as
        learning rate or num epochs
    :param metadata_args A list of arguments to be extracted from training_args
        and passed as metadata for the final, saved recipe.
    :param data_args: A list of arguments for dataset specification.
    :param teacher: teacher model for distillation. Set to 'self' to distill
        from the loaded model or 'disable' to turn of distillation
    :param kwargs: key word arguments passed to the parent class
    """

    def __init__(
        self,
        model: Module,
        model_state_path: str,
        recipe: str,
        recipe_args: Optional[Union[Dict[str, Any], str]] = None,
        metadata_args: Optional[List[str]] = None,
        data_args: Optional[List[str]] = None,
        teacher: Optional[Module] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            model_state_path=model_state_path,
            recipe=recipe,
            recipe_args=recipe_args,
            metadata_args=metadata_args,
            data_args=data_args,
            teacher=teacher,
            **kwargs,
        )


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
):
    """
    Post-processes the predictions of a question-answering model to convert them
    to answers that are substrings of the original contexts. This is the base
    postprocessing functions for models that only return start and end logits.

    :param examples: The non-preprocessed dataset. See main script for more
    :param features: The processed dataset. See main script for more
    :param predictions: The predictions of the model: two arrays containing the start
        logits and the end logits respectively. Its first dimension must match the
        number of elements of features
    :param version_2_with_negative: Whether or not the underlying dataset contains
        examples with no answers
    :param n_best_size: The total number of n-best predictions to generate when
        looking for an answer
    :param max_answer_length: The maximum length of an answer that can be generated.
        Needed because the start and end predictions are not conditioned on one another
    :param null_score_diff_threshold: The threshold used to select the null answer:
        if the best answer has a score that is less than the score of the null answer
        minus this threshold, the null answer is selected for this example. Only useful
        when version_2_with_negative is False
    :param output_dir: If provided, the dictionaries of predictions, n_best predictions
        (with their scores and logits) and, if version_2_with_negative=True, the
            dictionary of the scores differences between best and null answers, are
            saved in output_dir
    :param prefix: If provided, the dictionaries mentioned above are saved with
        prefix added to their names
    :param is_world_process_zero: Whether this process is the main process or not
        (used to determine if logging/saves should be done)
    :return: dictionary of prediction values
    """
    if len(predictions) != 2:
        raise ValueError(
            "`predictions` should be a tuple with two elements "
            "(start_logits, end_logits)."
        )
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Got {len(predictions[0])} predictions and {len(features)} features."
        )

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    _LOGGER.setLevel(log_level)
    _LOGGER.info(
        f"Post-processing {len(examples)} example predictions split into "
        f"{len(features)} features."
    )

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to
            # span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers
            # that do not have the maximum context available in the current feature.
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None
            )

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start
            # and end logits.
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices
                    # are out of bounds or correspond to part of the input_ids that
                    # are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # Don't consider answers with a length that is
                    # either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    # Don't consider answer that don't have the maximum context
                    # available (if such information is provided).
                    if (
                        token_is_max_context is not None
                        and not token_is_max_context.get(str(start_index), False)
                    ):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its
        # low score.
        if version_2_with_negative and not any(
            p["offsets"] == (0, 0) for p in predictions
        ):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we
        # create a fake prediction to avoid failure
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        # Compute the softmax of all scores (we do it with numpy to stay independent
        # from torch/tf in this file, using the LogSumExp trick)
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = (
                null_score
                - best_non_null_pred["start_logit"]
                - best_non_null_pred["end_logit"]
            )
            scores_diff_json[example["id"]] = float(
                score_diff
            )  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.float16, np.float32, np.float64))
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir,
            "predictions.json" if prefix is None else f"{prefix}_predictions.json",
        )
        nbest_file = os.path.join(
            output_dir,
            "nbest_predictions.json"
            if prefix is None
            else f"{prefix}_nbest_predictions.json",
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir,
                "null_odds.json" if prefix is None else f"{prefix}_null_odds.json",
            )

        _LOGGER.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        _LOGGER.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            _LOGGER.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions
