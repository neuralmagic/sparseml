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
Example of pruning a ResNet20-v1 model pretrained on the Cifar-10 dataset.
The pretrained model and this pruning script were adapted from:
https://keras.io/zh/integrations/cifar10_resnet/

Run the following command from the top repo directory:

   python3 integrations/keras/prune_resnet20.py

"""

import math
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sparseml.keras.optim import ScheduledModifierManager
from sparseml.keras.utils.callbacks import LossesAndMetricsLoggingCallback
from sparseml.keras.utils.exporter import ModelExporter
from sparseml.keras.utils.logger import TensorBoardLogger
from sparsezoo import Model


# Root directory
root_dir = "./integrations/keras"

# Logging setup
log_dir = os.path.join(root_dir, "tensorboard", "resnet20_v1")
log_dir += ":" + datetime.now().strftime("%Y%m%d-%H%M%S")
update_freq = 100  # Logging update every this many training steps

# Train/dataset setting
num_classes = 10
batch_size = 32
subtract_pixel_mean = True
data_augmentation = True

# Pruned model directory
pruned_model_dir = os.path.join(root_dir, "pruned")
if not os.path.isdir(pruned_model_dir):
    os.makedirs(pruned_model_dir)


def download_model_and_recipe(root_dir: str):
    """
    Download pretrained model and a pruning recipe
    """

    # Use the recipe stub
    recipe_file_path = (
        "zoo:cv/classification/resnet_v1-20/keras/sparseml/cifar_10/pruned-conservative"
    )

    # Load base model to prune
    base_zoo_model = Model(recipe_file_path)
    base_zoo_model.path = os.path.join(root_dir, "resnet20_v1")
    checkpoint = base_zoo_model.training.default
    model_file_path = checkpoint.get_file("model.h5").path
    recipe_file_path = base_zoo_model.recipes.default.path
    if not os.path.exists(model_file_path) or not model_file_path.endswith(".h5"):
        raise RuntimeError("Model file not found: {}".format(model_file_path))

    return model_file_path, recipe_file_path


def load_and_normalize_cifar10(subtract_pixel_mean: bool = True):
    """
    Load and normalize the Cifar-10 dataset
    """
    # Load the CIFAR10 data.
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize data.
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        X_train_mean = np.mean(X_train, axis=0)
        X_train -= X_train_mean
        X_test -= X_train_mean

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (X_train, y_train), (X_test, y_test)


def model_checkpoint_callback():
    """
    Create model checkpoint callback
    """
    pruned_model_name = "resnet20_v1.pruned.{epoch:03d}.h5"
    pruned_filepath = os.path.join(pruned_model_dir, pruned_model_name)

    # Prepare a callback for model saving
    checkpoint = ModelCheckpoint(
        filepath=pruned_filepath, monitor="val_accuracy", verbose=1, save_best_only=True
    )
    return checkpoint


def main():
    print("Load and normalize Cifar-10 dataset")
    (X_train, y_train), (X_test, y_test) = load_and_normalize_cifar10()

    model_file_path, recipe_file_path = download_model_and_recipe(root_dir)

    print("Load pretrained model")
    base_model = tf.keras.models.load_model(model_file_path)
    base_model.summary()

    scores = base_model.evaluate(X_test, y_test, verbose=1)
    print("Test loss:", scores[0])
    print("Test accuracy:", scores[1])

    optimizer = tf.keras.optimizers.Adam()

    steps_per_epoch = math.ceil(len(X_train) / batch_size)

    # Enhance the model and optimizer for pruning using the manager
    loggers = TensorBoardLogger(log_dir=log_dir, update_freq=update_freq)
    manager = ScheduledModifierManager.from_yaml(recipe_file_path)
    model_for_pruning, optimizer, callbacks = manager.modify(
        base_model, optimizer, steps_per_epoch, loggers=loggers
    )
    callbacks.append(LossesAndMetricsLoggingCallback(loggers))
    callbacks.append(model_checkpoint_callback())

    # Compile the enhanced model
    model_for_pruning.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=optimizer,
        metrics=["accuracy"],
        run_eagerly=True,
    )

    # Run training and pruning, with or without data augmentation.
    if not data_augmentation:
        print("Not using data augmentation.")
        model_for_pruning.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            shuffle=True,
            callbacks=callbacks,
            epochs=manager.max_epochs,
        )
    else:
        print("Using real-time data augmentation.")
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.0,
            # set range for random zoom
            zoom_range=0.0,
            # set range for random channel shifts
            channel_shift_range=0.0,
            # set mode for filling points outside the input boundaries
            fill_mode="nearest",
            # value used for fill_mode = "constant"
            cval=0.0,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0,
        )

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)

        # Fit the model on the batches generated by datagen.flow().
        model_for_pruning.fit_generator(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_test, y_test),
            epochs=manager.max_epochs,
            verbose=1,
            workers=4,
            callbacks=callbacks,
        )

    # Erase pruning masks and export to ONNX model
    pruned_model = manager.finalize(model_for_pruning)
    exporter = ModelExporter(pruned_model, output_dir=pruned_model_dir)
    onnx_model_name = "pruned_resnet20_v1.onnx"
    exporter.export_onnx(name=onnx_model_name)
    print(
        "Model exported to {}".format(os.path.join(pruned_model_dir, onnx_model_name))
    )


if __name__ == "__main__":
    main()
