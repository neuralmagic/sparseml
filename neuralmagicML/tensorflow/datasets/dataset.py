"""
General dataset implementations for TensorFlow
"""

from typing import Callable, Iterable, Tuple, Any, List
from abc import abstractmethod, ABCMeta
import os
import glob
import random
import numpy

from neuralmagicML.utils import clean_path
from neuralmagicML.tensorflow.utils import tf_compat


__all__ = [
    "random_scaling_crop",
    "center_square_crop",
    "create_split_iterators_handle",
    "Dataset",
    "ImageFolderDataset",
]


def random_scaling_crop(
    scale_range: Tuple[int, int] = (0.08, 1.0),
    ratio_range: Tuple[int, int] = (3.0 / 4.0, 4.0 / 3.0),
):
    """
    Random crop implementation which also randomly scales the crop taken
    as well as the aspect ratio of the crop.

    :param scale_range: the (min, max) of the crop scales to take from the orig image
    :param ratio_range: the (min, max) of the aspect ratios to take from the orig image
    :return: a callable to randomly crop a passed in image tensor
    """

    def rand_crop(img: tf_compat.Tensor):
        orig_shape = tf_compat.shape(img)
        scale = tf_compat.random.uniform(
            shape=[1], minval=scale_range[0], maxval=scale_range[1]
        )[0]
        ratio = tf_compat.random.uniform(
            shape=[1], minval=ratio_range[0], maxval=ratio_range[1]
        )[0]
        height = tf_compat.math.minimum(
            tf_compat.cast(
                tf_compat.round(
                    tf_compat.cast(orig_shape[0], dtype=tf_compat.float32)
                    * scale
                    / ratio
                ),
                tf_compat.int32,
            ),
            orig_shape[0],
        )
        width = tf_compat.math.minimum(
            tf_compat.cast(
                tf_compat.round(
                    tf_compat.cast(orig_shape[1], dtype=tf_compat.float32) * scale
                ),
                tf_compat.int32,
            ),
            orig_shape[1],
        )
        img = tf_compat.image.random_crop(img, [height, width, orig_shape[2]])

        return img

    return rand_crop


def center_square_crop():
    """
    Take a square crop centered in the a image

    :return: a callable to take the square crop from a passed in image tensor
    """
    # TODO: add support for padding the outside of the image before taking the crop
    # will better match PyTorch impl

    def cent_crop(img: tf_compat.Tensor):
        orig_shape = tf_compat.shape(img)
        min_size = tf_compat.cond(
            tf_compat.greater_equal(orig_shape[0], orig_shape[1]),
            lambda: orig_shape[1],
            lambda: orig_shape[0],
        )
        padding_height = tf_compat.cast(
            tf_compat.round(
                tf_compat.div(
                    tf_compat.cast(
                        tf_compat.subtract(orig_shape[0], min_size), tf_compat.float32
                    ),
                    2.0,
                )
            ),
            tf_compat.int32,
        )
        padding_width = tf_compat.cast(
            tf_compat.round(
                tf_compat.div(
                    tf_compat.cast(
                        tf_compat.subtract(orig_shape[1], min_size), tf_compat.float32
                    ),
                    2.0,
                )
            ),
            tf_compat.int32,
        )
        img = tf_compat.image.crop_to_bounding_box(
            img, padding_height, padding_width, min_size, min_size
        )

        return img

    return cent_crop


def create_split_iterators_handle(split_datasets: Iterable) -> Tuple[Any, Any, List]:
    """
    Create an iterators handle for switching between datasets easily while training.

    :param split_datasets: the datasets to create the splits and handle for
    :return: a tuple containing the handle that should be set with a feed dict,
        the iterator used to get the next batch,
        and a list of the iterators created from the split_datasets
    """
    output_types = None
    output_shapes = None
    split_iterators = []

    # Get correct TF data functions for backwards compatibility with TF
    get_output_types = (
        tf_compat.data.get_output_types
        if hasattr(tf_compat.data, "get_output_types")
        else lambda dataset: dataset.output_types
    )
    get_output_shapes = (
        tf_compat.data.get_output_shapes
        if hasattr(tf_compat.data, "get_output_shapes")
        else lambda dataset: dataset.output_shapes
    )
    make_initializable_iterator = (
        tf_compat.data.make_initializable_iterator
        if hasattr(tf_compat.data, "make_initializable_iterator")
        else lambda dataset: dataset.make_initializable_iterator()
    )

    for split_dataset in split_datasets:
        output_types = get_output_types(split_dataset)
        output_shapes = get_output_shapes(split_dataset)
        split_iterators.append(make_initializable_iterator(split_dataset))

    handle = tf_compat.placeholder(tf_compat.string, shape=[])
    iterator = tf_compat.data.Iterator.from_string_handle(
        handle, output_types, output_shapes
    )

    return handle, iterator, split_iterators


class Dataset(metaclass=ABCMeta):
    """
    Generic dataset implementation for TensorFlow.
    Expected to work with the tf.data APIs
    """

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    def build(
        self,
        batch_size: int,
        repeat_count: int = None,
        shuffle_buffer_size: int = None,
        prefetch_buffer_size: int = None,
        num_parallel_calls: int = None,
    ):
        """
        Create the dataset in the current graph using tf.data APIs

        :param batch_size: the batch size to create the dataset for
        :param repeat_count: the number of times to repeat the dataset,
            if unset or None, will repeat indefinitely
        :param shuffle_buffer_size: None if not shuffling,
            otherwise the size of the buffer to use for shuffling data
        :param prefetch_buffer_size: None if not prefetching,
            otherwise the size of the buffer to use for buffering
        :param num_parallel_calls: the number of parallel calls to run the
            processor function with
        :return: a tf.data.Dataset instance
        """
        with tf_compat.name_scope(self.name_scope()):
            dataset = self.creator()

            if shuffle_buffer_size and shuffle_buffer_size > 0:
                dataset = dataset.shuffle(
                    shuffle_buffer_size, reshuffle_each_iteration=True
                )

            dataset = dataset.map(self.processor, num_parallel_calls=num_parallel_calls)
            dataset = dataset.batch(batch_size)
            dataset = dataset.repeat(repeat_count)

            if prefetch_buffer_size and prefetch_buffer_size > 0:
                dataset = dataset.prefetch(prefetch_buffer_size)

        return dataset

    @abstractmethod
    def creator(self):
        """
        Implemented by sub classes to create a tf.data dataset for the given impl.

        :return: a created tf.data dataset
        """
        raise NotImplementedError()

    @abstractmethod
    def processor(self, *args, **kwargs):
        """
        Implemented by sub classes to parallelize and map processing functions
        for loading the data of the dataset into memory.

        :param args: generic inputs for processing
        :param kwargs: generic inputs for processing
        :return: the processed tensors
        """
        raise NotImplementedError()

    @abstractmethod
    def name_scope(self) -> str:
        """
        Implemented by sub classes to get a name scope for building the dataset
        in the graph

        :return: the name scope the dataset should be built under in the graph
        """
        raise NotImplementedError()


class ImageFolderDataset(Dataset):
    """
    Implementation for loading an image folder structure into a dataset.
    |Image folders should be of the form:
    |    - root
    |        - label
    |            - images

    :param root: the root location for the dataset's images to load
    :param image_size: the size of the image to reshape to
    :param transforms: any additional transforms to apply to the images
    :param normalizer: the function to normalize images
    """

    def __init__(
        self,
        root: str,
        image_size: int = None,
        transforms: Iterable[Callable] = None,
        normalizer: Callable = None,
    ):
        self._root = clean_path(root)
        self._image_size = image_size
        self._transforms = transforms
        self._normalizer = normalizer
        self._num_examples = None
        self._length = None

        if not os.path.exists(self._root):
            raise ValueError("root must exist")

    def __len__(self):
        if self._length is None:
            self._length = len(
                [None for _ in glob.glob(os.path.join(self.root, "*", "*"))]
            )

        return self._length

    @property
    def root(self) -> str:
        """
        :return: the root location for the dataset's images to load
        """
        return self._root

    @property
    def image_size(self) -> int:
        """
        :return: the size of the images to resize to
        """
        return self._image_size

    @property
    def transforms(self) -> Iterable[Callable]:
        """
        :return: any additional transforms to apply to the images
        """
        return self._transforms

    @property
    def normalizer(self) -> Callable:
        """
        :return: the normalizer to apply to the images
        """
        return self._normalizer

    def processor(self, file_path: tf_compat.Tensor, label: tf_compat.Tensor):
        """
        :param file_path: the path to the file to load an image from
        :param label: the label for the given image
        :return: a tuple containing the processed image and label
        """
        with tf_compat.name_scope("img_to_tensor"):
            img = tf_compat.read_file(file_path)
            img = tf_compat.image.decode_image(img)
            img = tf_compat.cast(img, dtype=tf_compat.float32)

        if self._transforms:
            with tf_compat.name_scope("transforms"):
                for index, trans in enumerate(self._transforms):
                    with tf_compat.name_scope(str(index)):
                        img = trans(img)

        if self._image_size is not None:
            with tf_compat.name_scope("resize"):
                try:
                    img = tf_compat.image.resize(
                        img, [self._image_size, self._image_size]
                    )
                except Exception:
                    img = tf_compat.image.resize_images(
                        img, [self._image_size, self._image_size]
                    )

        if self._normalizer is not None:
            with tf_compat.name_scope("normalize"):
                img = self._normalizer(img)

        return img, label

    def creator(self):
        """
        :return: a created dataset that gives the file_path and label for each
            image under self.root
        """
        labels_strs = [
            file.split(os.path.sep)[-1]
            for file in glob.glob(os.path.join(self.root, "*"))
        ]
        labels_strs.sort()
        labels_dict = {
            lab: numpy.identity(len(labels_strs))[index].tolist()
            for index, lab in enumerate(labels_strs)
        }
        files_labels = [
            (file, labels_dict[file.split(os.path.sep)[-2]])
            for file in glob.glob(os.path.join(self.root, "*", "*"))
        ]
        random.shuffle(files_labels)
        files, labels = zip(*files_labels)
        files = tf_compat.constant(files)
        labels = tf_compat.constant(labels)

        return tf_compat.data.Dataset.from_tensor_slices((files, labels))

    def name_scope(self) -> str:
        """
        :return: the name scope the dataset should be built under in the graph
        """
        return "ImageFolderDataset_{}".format(self.root.replace(os.path.sep, "."))
