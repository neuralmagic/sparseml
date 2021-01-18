"""
Example of pruning a ResNet20-v1 model pretrained on the Cifar-10 dataset.

Two artifacts provided to be used in this example:
- examples/keras/models/ResNet20-v1.h5: the pretrained ResNet20-v1 model with 91.85% validation accuracy on Cifar-10 
- examples/keras/configs/prune_resnet20_10epochs.yaml: an example pruning schedule file

Run the following command from the top repo directory:

   python3 examples/keras/prune_resnet20.py

"""

from datetime import datetime
import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
import numpy as np

from sparseml.keras.optim import ScheduledModifierManager
from sparseml.keras.utils import TensorBoardLogger

root_dir = "./examples/keras"
yaml_file_name = "prune_resnet20_10epochs"
yaml_file_path = os.path.join(root_dir, "configs", "{}.yaml".format(yaml_file_name))

model_dir = "{}/models".format(root_dir)
model_name = "ResNet20-v1.h5"

log_dir = os.path.join(model_dir, "tensorboard", yaml_file_name)
log_dir += "--" + datetime.now().strftime("%Y%m%d-%H%M%S")

update_freq = 100  # Logging update every this many training steps
num_classes = 10
batch_size = 32
subtract_pixel_mean = True
data_augmentation = True

epochs = 10

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

print("x_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")
print("y_train shape:", y_train.shape)

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Prepare model model saving directory.
pruned_model_dir = os.path.join(model_dir, "pruned")
pruned_model_name = "%s.pruned.{epoch:03d}.h5" % model_name
if not os.path.isdir(pruned_model_dir):
    os.makedirs(pruned_model_dir)
pruned_filepath = os.path.join(pruned_model_dir, pruned_model_name)

# Prepare a callback for model saving
checkpoint = ModelCheckpoint(
    filepath=pruned_filepath, monitor="val_accuracy", verbose=1, save_best_only=True
)


def main():
    print("Load pretrained model")

    base_model = tf.keras.models.load_model(os.path.join(model_dir, model_name))
    base_model.summary()

    scores = base_model.evaluate(X_test, y_test, verbose=1)
    print("Test loss:", scores[0])
    print("Test accuracy:", scores[1])

    optimizer = tf.keras.optimizers.Adam()

    steps_per_epoch = math.ceil(len(X_train) / batch_size)

    # Enhance the model and optimizer for pruning using the manager
    loggers = TensorBoardLogger(log_dir=log_dir, update_freq=update_freq)
    manager = ScheduledModifierManager.from_yaml(yaml_file_path)
    model_for_pruning, optimizer, callbacks = manager.modify(
        base_model, optimizer, steps_per_epoch, loggers=loggers
    )
    callbacks.append(checkpoint)

    # Compile the enhanced model
    model_for_pruning.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer="adam",
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
            epochs=epochs,
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
            epochs=epochs,
            verbose=1,
            workers=4,
            callbacks=callbacks,
        )


if __name__ == "__main__":
    main()
