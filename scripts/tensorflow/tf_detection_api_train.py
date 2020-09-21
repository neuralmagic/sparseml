"""
Object detection recal script. Setup to support the following use cases:
- training object detection architectures
- pruning object detection architectures
- transfer learning object detection architectures
- evaluating object detection architectures


##########
Command help:
usage: classification_train.py [-h] --model-dir MODEL_DIR
                                --pipeline-config-path PIPELINE_PATH
                                [--recal-config-path RECAL_CONFIG_PATH]
                                --num-train-steps TRAIN_STEPS
                                --num-steps-per-epoch STEPS_PER_EPOCH
                                [--eval-training-data EVAL_TRAIN_DAT]
                                [--sample-1-of-n-eval-examples SAMPLE_1_OF_N_EVAL]
                                [--sample-1-of-n-eval-on-train-examples SAMPLE_1_OF_N_TRAIN]
                                [--hparams-overrides HPARAMS_OVERRIDES]
                                [--checkpoint-dir CHECKPOINT_DIR]
                                [--run-once RUN_ONCE]

###########
Example command fine-tuning a Faster-RCNN Resnet101 model on VOC dataset. The configuration
file faster_rcnn_resnet101_voc07.config defines model architecture, checkpoint, training and eval
config. It also includes the location of the dataset. The recal.yaml defines learning rate modifiers
to be applied.

python scripts/tensorflow/tf_detection_api_train.py \
    --model-dir my_models \
    --pipeline-config-path my_config/faster_rcnn_resnet101_voc07.config \
    --num-train-steps 50000 --num-steps-per-epoch 1000 \
    --recal-config-path my_config/recal.yaml

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from object_detection import model_hparams
from object_detection import model_lib

from neuralmagicML.tensorflow.recal import ScheduledModifierManager
from neuralmagicML.tensorflow.utils import tf_compat


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and/or prune an object_detection "
        "architecture on a dataset"
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to output model directory "
        "where event and checkpoint files will be written.",
    )
    parser.add_argument(
        "--pipeline-config-path",
        type=str,
        required=True,
        help="Path to pipeline config file.",
    )
    parser.add_argument(
        "--recal-config-path",
        type=str,
        required=False,
        help="Path to recalibration config file.",
    )
    parser.add_argument(
        "--num-train-steps", type=int, required=True, help="Number of train steps."
    )
    parser.add_argument(
        "--num-steps-per-epoch",
        type=int,
        required=True,
        help="Number of steps per traininng epoch.",
    )
    parser.add_argument(
        "--eval-training-data",
        type=bool,
        required=False,
        help="If training data should be evaluated for this job. Note "
        "that one call only use this in eval-only mode, and "
        "`checkpoint_dir` must be supplied.",
    )
    parser.add_argument(
        "--sample-1-of-n-eval-examples",
        type=int,
        default=1,
        help="Will sample one of every n eval input examples, where n is provided.",
    )
    parser.add_argument(
        "--sample-1-of-n-eval-on-train-examples",
        type=int,
        default=5,
        help="Will sample "
        "one of every n train input examples for evaluation, "
        "where n is provided. This is only used if "
        "`eval_training_data` is True.",
    )
    parser.add_argument(
        "--hparams-overrides",
        default=None,
        help="Hyperparameter overrides, "
        "represented as a string containing comma-separated "
        "hparam_name=value pairs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to directory holding a checkpoint.  If "
        "`checkpoint_dir` is provided, this binary operates in eval-only mode, "
        "writing resulting metrics to `model_dir`.",
    )
    parser.add_argument(
        "--run-once",
        type=bool,
        default=False,
        help="If running in eval-only mode, whether to run just "
        "one round of eval vs running continuously (default).",
    )
    return parser.parse_args()


def main(args):

    config = tf_compat.estimator.RunConfig(model_dir=args.model_dir)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(args.hparams_overrides),
        pipeline_config_path=args.pipeline_config_path,
        train_steps=args.num_train_steps,
        sample_1_of_n_eval_examples=args.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(
            args.sample_1_of_n_eval_on_train_examples
        ),
    )
    estimator = train_and_eval_dict["estimator"]
    train_input_fn = train_and_eval_dict["train_input_fn"]
    eval_input_fns = train_and_eval_dict["eval_input_fns"]
    if len(eval_input_fns) > 1:
        raise ValueError("Currently only one evaluation specification is supported")
    eval_on_train_input_fn = train_and_eval_dict["eval_on_train_input_fn"]
    predict_input_fn = train_and_eval_dict["predict_input_fn"]
    train_steps = train_and_eval_dict["train_steps"]

    manager = ScheduledModifierManager.from_yaml(args.recal_config_path)
    manager.modify_estimator(estimator, args.num_steps_per_epoch)

    if args.checkpoint_dir:
        if args.eval_training_data:
            name = "training_data"
            input_fn = eval_on_train_input_fn
        else:
            name = "validation_data"
            input_fn = eval_input_fns[0]
        if args.run_once:
            estimator.evaluate(
                input_fn,
                steps=None,
                checkpoint_path=tf_compat.train.latest_checkpoint(args.checkpoint_dir),
            )
        else:
            model_lib.continuous_eval(
                estimator, args.checkpoint_dir, input_fn, train_steps, name
            )
    else:
        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fns,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_on_train_data=False,
        )
        tf_compat.estimator.train_and_evaluate(
            estimator,
            train_spec,
            tf_compat.estimator.EvalSpec(
                input_fn=eval_specs[0].input_fn,
                steps=eval_specs[0].steps,
                hooks=eval_specs[0].hooks,
                exporters=eval_specs[0].exporters,
                start_delay_secs=120,
                throttle_secs=1800,
            ),
        )


if __name__ == "__main__":
    _args = parse_args()
    main(_args)
