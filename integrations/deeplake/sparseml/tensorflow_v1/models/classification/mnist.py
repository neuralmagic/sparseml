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

from sparseml.tensorflow_v1.models.estimator import ClassificationEstimatorModelFn
from sparseml.tensorflow_v1.models.registry import ModelRegistry
from sparseml.tensorflow_v1.nn import activation, conv2d, fc
from sparseml.tensorflow_v1.utils import tf_compat


__all__ = ["mnist_net"]


BASE_NAME_SCOPE = "mnist_net"


@ModelRegistry.register(
    key=["mnistnet"],
    input_shape=(28, 28, 1),
    domain="cv",
    sub_domain="classification",
    architecture="mnistnet",
    sub_architecture="none",
    default_dataset="mnist",
    default_desc="base",
    default_model_fn_creator=ClassificationEstimatorModelFn,
    base_name_scope=BASE_NAME_SCOPE,
    tl_ignore_tens=[],
)
def mnist_net(
    inputs: tf_compat.Tensor, num_classes: int = 10, act: str = None
) -> tf_compat.Tensor:
    """
    A simple convolutional model created for the MNIST dataset

    :param inputs: the inputs tensor to create the network for
    :param num_classes: the number of classes to create the final layer for
    :param act: the final activation to use in the model,
        supported: [None, relu, sigmoid, softmax]
    :return: the logits output from the created network
    """
    if act not in [None, "sigmoid", "softmax"]:
        raise ValueError("unsupported value for act given of {}".format(act))

    with tf_compat.variable_scope(BASE_NAME_SCOPE, reuse=tf_compat.AUTO_REUSE):
        with tf_compat.variable_scope("blocks", reuse=tf_compat.AUTO_REUSE):
            x_tens = conv2d(
                name="conv0",
                x_tens=inputs,
                in_chan=1,
                out_chan=16,
                kernel=5,
                stride=1,
                padding="SAME",
                act="relu",
            )
            x_tens = conv2d(
                name="conv1",
                x_tens=x_tens,
                in_chan=16,
                out_chan=32,
                kernel=5,
                stride=2,
                padding="SAME",
                act="relu",
            )
            x_tens = conv2d(
                name="conv2",
                x_tens=x_tens,
                in_chan=32,
                out_chan=64,
                kernel=5,
                stride=1,
                padding="SAME",
                act="relu",
            )
            x_tens = conv2d(
                name="conv3",
                x_tens=x_tens,
                in_chan=64,
                out_chan=128,
                kernel=5,
                stride=2,
                padding="SAME",
                act="relu",
            )

        with tf_compat.variable_scope("classifier"):
            x_tens = tf_compat.reduce_mean(x_tens, axis=[1, 2])
            x_tens = tf_compat.reshape(x_tens, [-1, 128])
            x_tens = fc(name="fc", x_tens=x_tens, in_chan=128, out_chan=num_classes)

        with tf_compat.variable_scope("logits"):
            logits = activation(x_tens, act)

    return logits
