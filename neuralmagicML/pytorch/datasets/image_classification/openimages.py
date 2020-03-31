"""
OpenImages dataset implementations for the image classification field in
computer vision. More info for the dataset can be found
`here <https://opensource.google/projects/open-images-dataset>`__.
"""

from typing import Iterator, Tuple, Dict, List, Union
import os
import errno
from tqdm import tqdm
import csv
import multiprocessing
from PIL import Image
import pandas
import mmap
import json
import sys
from threading import Lock

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from neuralmagicML.pytorch.datasets.registry import DatasetRegistry
from neuralmagicML.pytorch.utils import MultiDownloader, DownloadResult, ParallelWorker


__all__ = ["OpenImagesDataset"]


_CLASS_NAMES = "https://storage.googleapis.com/openimages/v5/class-descriptions.csv"
_TRAINABLE_CLASSES = (
    "https://storage.googleapis.com/openimages/v5/classes-trainable.txt"
)
_TRAIN_IMAGE_IDS = (
    "https://storage.googleapis.com/openimages/v5/"
    "train-images-with-labels-with-rotation.csv"
)
_TRAIN_HUMAN_LABELS = (
    "https://storage.googleapis.com/openimages/v5/"
    "train-annotations-human-imagelabels.csv"
)
_TRAIN_MACHINE_LABELS = (
    "https://storage.googleapis.com/openimages/v5/"
    "train-annotations-machine-imagelabels.csv"
)

_VAL_IMAGE_IDS = (
    "https://storage.googleapis.com/openimages/2018_04/"
    "validation/validation-images-with-rotation.csv"
)
_VAL_HUMAN_LABELS = (
    "https://storage.googleapis.com/openimages/v5/"
    "validation-annotations-human-imagelabels.csv"
)
_VAL_MACHINE_LABELS = (
    "https://storage.googleapis.com/openimages/v5/"
    "validation-annotations-machine-imagelabels.csv"
)

_TEST_IMAGE_IDS = (
    "https://storage.googleapis.com/openimages/2018_04/"
    "test/test-images-with-rotation.csv"
)
_TEST_HUMAN_LABELS = (
    "https://storage.googleapis.com/openimages/v5/"
    "test-annotations-human-imagelabels.csv"
)
_TEST_MACHINE_LABELS = (
    "https://storage.googleapis.com/openimages/v5/"
    "test-annotations-machine-imagelabels.csv"
)

_RGB_MEANS = [0.485, 0.456, 0.406]
_RGB_STDS = [0.229, 0.224, 0.225]


@DatasetRegistry.register(
    key=["openimages", "open_images"],
    attributes={
        "num_classes": 8658,
        "transform_means": _RGB_MEANS,
        "transform_stds": _RGB_STDS,
    },
)
class OpenImagesDataset(Dataset):
    """
    Implementation for the Open Images dataset. Targeted at v5.

    :param root: The root folder to find the dataset at,
        if not found will download here if download=True
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param confidence: The confidence level needed from machine labeled images
        to include in the dataset
    :param image_size: The image size to output from the dataset
    :param samples_num_workers: The number of CPU workers to use for creating data
    :param download_num_workers: The number of CPU workers to use for downloading data
    :param download_sample_size: If > 0, then will only download this number of images
        instead of the full dataset
    :param download_image_size: The size of the images to store locally on the disk
        after downloading
    :param download_test_images: True to download the test images, False otherwise.
        Default is False. (train and validation are automatically downloaded)
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        rand_trans: bool = False,
        confidence: float = 1.0,
        image_size: int = 224,
        samples_num_workers: int = 0,
        download_num_workers: int = 0,
        download_sample_size: int = -1,
        download_image_size: int = 600,
        download_test_images: bool = False,
    ):
        print("OpenImagesDataset: checking download")
        OpenImagesDataset._download(
            root,
            download_num_workers,
            download_sample_size,
            download_image_size,
            download_test_images,
        )

        self.root = root

        if rand_trans:
            trans = [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            resize_scale = 256.0 / 224.0  # standard used
            trans = [
                transforms.Resize(round(resize_scale * image_size)),
                transforms.CenterCrop(image_size),
            ]

        trans.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=_RGB_MEANS, std=_RGB_STDS),
            ]
        )

        self.transform = transforms.Compose(trans)
        self.confidence = confidence
        self.image_size = image_size

        if samples_num_workers <= 0:
            samples_num_workers = multiprocessing.cpu_count()

        split = "train" if train else "validation"
        print("OpenImagesDataset: collecting samples for {}".format(split))
        self.samples = OpenImagesDataset._get_split_samples(
            split, root, confidence, samples_num_workers
        )

        self.trainable_classes_lock = Lock()
        self.trainable_classes = pandas.read_csv(
            os.path.join(
                root, "meta", os.path.basename(OpenImagesDataset.TRAINABLE_CLASSES)
            ),
            names=["LabelName"],
        )

    def __getitem__(self, index):
        x_path, y_human_path, y_machine_path = self.samples[index]
        x_feat = default_loader(x_path)

        if self.transform is not None:
            x_feat = self.transform(x_feat)

        with self.trainable_classes_lock:
            label_names = (
                self.trainable_classes.copy()["LabelName"].sort_values().tolist()
            )
            confidence = self.confidence

        y_lab = [0.0] * len(label_names)
        human_only = confidence >= (1.0 - sys.float_info.epsilon) or confidence < 0.0

        if y_machine_path is not None and not human_only:
            with open(y_machine_path, "r") as y_machine_file:
                y_machine_labels = json.load(y_machine_file)

            for lab, val in y_machine_labels.items():
                val = float(val)

                if val < confidence or lab not in label_names:
                    continue

                index = label_names.index(lab)
                y_lab[index] = 1.0

        if y_human_path is not None:
            with open(y_human_path, "r") as y_human_file:
                y_human_labels = json.load(y_human_file)

            for lab, val in y_human_labels.items():
                val = 1.0 if float(val) > 0.5 else 0.0

                if lab not in label_names:
                    continue

                index = label_names.index(lab)
                y_lab[index] = val

        y_lab = torch.FloatTensor(y_lab)

        return x_feat, y_lab

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )

        return fmt_str

    @staticmethod
    def _get_split_samples(
        split: str, root: str, confidence: float, num_workers: int
    ) -> List[Tuple[str, str, str]]:
        samples_path = os.path.join(root, split)

        def _files_generator() -> Iterator[Tuple[str, float]]:
            for _, __, _files in os.walk(samples_path):
                for _file in _files:
                    yield os.path.join(samples_path, _file), confidence

        def _samples_worker(
            _vals: Tuple[str, float]
        ) -> Union[None, Tuple[str, str, str]]:
            _file_path, _confidence = _vals
            _human_only = (
                _confidence >= (1.0 - sys.float_info.epsilon) or _confidence < 0.0
            )
            _check_file_suffix = "lab-human.json" if _human_only else "lab-machine.json"

            if not _file_path.endswith(_check_file_suffix):
                return None

            _samples_path = os.path.dirname(_file_path)
            _image_id = os.path.basename(_file_path).split(".")[0]
            _image_path = os.path.join(_samples_path, "{}.jpg".format(_image_id))
            _human_path = os.path.join(
                _samples_path, "{}.lab-human.json".format(_image_id)
            )
            _machine_path = os.path.join(
                _samples_path, "{}.lab-machine.json".format(_image_id)
            )

            if not os.path.exists(_image_path):
                _image_path = None

            if not os.path.exists(_human_path):
                _human_path = None

            if not os.path.exists(_machine_path):
                _machine_path = None

            if _image_path is not None and (
                (_machine_path is not None and not _human_only)
                or _human_path is not None
            ):
                return _image_path, _human_path, _machine_path

            return None

        print("collecting data samples for {}...".format(split))
        worker = ParallelWorker(
            _samples_worker, num_workers, indefinite=True, max_source_size=1000
        )
        worker.add_async_generator(_files_generator())
        worker.start()

        samples = []
        num_files = len(os.listdir(samples_path))

        for vals in tqdm(
            worker,
            desc="collecting data samples for {}...".format(split),
            total=num_files,
        ):
            if vals is None:
                continue

            samples.append(vals)

        return samples

    @staticmethod
    def _download(
        root_dir: str,
        num_workers: int,
        sample_size: int,
        image_size: int,
        test_images: bool,
    ):
        try:
            os.makedirs(root_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                # Unexpected OSError, re-raise.
                raise

        downloaded_file_path = os.path.join(root_dir, ".downloaded")

        if os.path.exists(downloaded_file_path):
            print(
                (
                    "Open Images already downloaded in {} , "
                    "delete the folder to redownload"
                ).format(root_dir)
            )

            return

        meta_dir = os.path.join(root_dir, "meta")

        try:
            os.makedirs(meta_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                # Unexpected OSError, re-raise.
                raise

        OpenImagesDataset._download_supporting_files(meta_dir, num_workers)
        OpenImagesDataset._download_images(
            root_dir, meta_dir, num_workers, sample_size, image_size, test_images
        )

        with open(downloaded_file_path, "wb") as downloaded_file:
            pass

    @staticmethod
    def _download_supporting_files(meta_dir: str, num_workers: int):
        supporting_sources = [
            _CLASS_NAMES,
            _TRAINABLE_CLASSES,
            _TRAIN_HUMAN_LABELS,
            _TRAIN_MACHINE_LABELS,
            _TRAIN_IMAGE_IDS,
            _VAL_HUMAN_LABELS,
            _VAL_MACHINE_LABELS,
            _VAL_IMAGE_IDS,
            _TEST_HUMAN_LABELS,
            _TEST_MACHINE_LABELS,
            _TEST_IMAGE_IDS,
        ]
        source_dests = []

        for source in supporting_sources:
            source_dests.append(
                (
                    os.path.basename(source),
                    source,
                    os.path.join(meta_dir, os.path.basename(source)),
                )
            )

        downloader = MultiDownloader(source_dests, num_workers=num_workers)

        for download in tqdm(downloader, desc="downloading supporting files"):
            if download.err:
                raise RuntimeError(
                    "Exception while downloading {} from {} to {} : {}".format(
                        os.path.basename(download.source),
                        download.source,
                        download.dest,
                        download.err,
                    )
                )

            print(
                "\nDownloaded {} from {} to {}".format(
                    os.path.basename(download.source), download.source, download.dest
                )
            )

    @staticmethod
    def _download_images(
        root_dir: str,
        meta_dir: str,
        num_workers: int,
        sample_size: int,
        image_size: int,
        test_images: bool,
    ):
        OpenImagesDataset._download_split_images(
            "validation",
            os.path.join(meta_dir, os.path.basename(_VAL_IMAGE_IDS)),
            os.path.join(meta_dir, os.path.basename(_VAL_HUMAN_LABELS)),
            os.path.join(meta_dir, os.path.basename(_VAL_MACHINE_LABELS)),
            root_dir,
            num_workers,
            sample_size,
            image_size,
        )
        OpenImagesDataset._download_split_images(
            "train",
            os.path.join(meta_dir, os.path.basename(_TRAIN_IMAGE_IDS)),
            os.path.join(meta_dir, os.path.basename(_TRAIN_HUMAN_LABELS)),
            os.path.join(meta_dir, os.path.basename(_TRAIN_MACHINE_LABELS)),
            root_dir,
            num_workers,
            sample_size,
            image_size,
        )

        if test_images:
            OpenImagesDataset._download_split_images(
                "test",
                os.path.join(meta_dir, os.path.basename(_TEST_IMAGE_IDS)),
                os.path.join(meta_dir, os.path.basename(_TEST_HUMAN_LABELS)),
                os.path.join(meta_dir, os.path.basename(_TEST_MACHINE_LABELS)),
                root_dir,
                num_workers,
                sample_size,
                image_size,
            )

    @staticmethod
    def _download_split_images(
        split_name: str,
        meta_path: str,
        human_labels_path: str,
        machine_labels_path: str,
        root_dir: str,
        num_workers: int,
        sample_size: int,
        image_size: int,
    ):
        print("starting download for {}".format(split_name))
        images_dir = os.path.join(root_dir, split_name)
        downloader = _ImageDownloader(
            split_name, images_dir, meta_path, num_workers, sample_size
        )

        for download in downloader.download():
            pass

        labeler = _ImageLabeler(
            split_name,
            images_dir,
            human_labels_path,
            machine_labels_path,
            num_workers,
            image_size,
        )
        labeler.label()


class _ImageDownloader(object):
    def __init__(
        self,
        split_name: str,
        images_dir: str,
        meta_path: str,
        num_workers: int = 0,
        sample_size: int = -1,
        image_size: int = 600,
    ):
        self._split_name = split_name
        self._images_dir = images_dir
        self._meta_path = meta_path
        self._num_workers = num_workers
        self._sample_size = sample_size
        self._image_size = image_size

    def download(self) -> Iterator[DownloadResult]:
        try:
            os.makedirs(self._images_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                # Unexpected OSError, re-raise.
                raise

        source_dests = []
        print("collecting {} images for download...".format(self._split_name))

        with open(self._meta_path) as meta_file:
            meta_csv = csv.reader(meta_file, delimiter=",")
            header = None

            for row in meta_csv:
                if header is None:
                    header = row
                    continue

                image_id = row[0]
                image_url = row[2]
                source_dests.append(
                    (
                        image_id,
                        image_url,
                        os.path.join(self._images_dir, "{}.jpg".format(image_id)),
                    )
                )

                if len(source_dests) >= self._sample_size > 0:
                    break

        downloader = MultiDownloader(
            source_dests,
            downloaded_callback=self._resize,
            num_workers=self._num_workers,
        )

        for download in tqdm(
            downloader, desc="downloading {} images".format(self._split_name)
        ):
            yield download

    def _resize(self, download: DownloadResult):
        if download.err is not None:
            return

        try:
            image = Image.open(download.dest)

            if min(image.size) > self._image_size:
                x_to_y = float(image.size[0]) / float(image.size[1])
                new_x = (
                    self._image_size
                    if x_to_y <= 1.0
                    else round(self._image_size * x_to_y)
                )
                new_y = (
                    self._image_size
                    if x_to_y > 1.0
                    else round(self._image_size / x_to_y)
                )
                resized = image.resize((new_x, new_y), Image.ANTIALIAS)
                resized.save(download.dest)
        except Exception as err:
            download.err = err

            return


class _ImageLabeler(object):
    def __init__(
        self,
        split_name: str,
        images_dir: str,
        human_labels_path: str,
        machine_labels_path: str,
        num_workers: int = 0,
        image_size: int = 600,
    ):
        if num_workers < 1:
            num_workers = (
                multiprocessing.cpu_count()
            )  # scale with the number of cores on the machine

        self._num_workers = num_workers
        self._split_name = split_name
        self._images_dir = images_dir
        self._human_labels_path = human_labels_path
        self._machine_labels_path = machine_labels_path
        self._image_size = image_size

    def label(self):
        def get_num_lines(fp):
            buf = mmap.mmap(fp.fileno(), 0)
            lines = 0
            while buf.readline():
                lines += 1
            return lines

        labels = [
            ("machine", self._machine_labels_path),
            ("human", self._human_labels_path),
        ]

        for (labels_type, labels_path) in labels:
            print("parsing {} labels for {}...".format(labels_type, self._split_name))
            label_worker = ParallelWorker(
                _ImageLabeler._create_label,
                self._num_workers,
                indefinite=True,
                max_source_size=100,
            )
            label_worker.start()

            with open(labels_path, "r+") as labels_file:
                labels_reader = csv.reader(labels_file)
                header = None
                image_id = None
                labels = None

                for line in tqdm(
                    labels_reader,
                    desc="creating {} labels for {}...".format(
                        labels_type, self._split_name
                    ),
                    total=get_num_lines(labels_file),
                ):
                    if header is None:
                        header = line
                        continue

                    if image_id != line[0]:
                        if image_id is not None:
                            label_worker.add_item(
                                (image_id, labels_type, labels, self._images_dir)
                            )

                        image_id = line[0]
                        labels = {}
                    else:
                        labels[line[2]] = float(line[3])

                if len(labels) > 0:
                    label_worker.add_item(
                        (image_id, labels_type, labels, self._images_dir)
                    )

            label_worker.indefinite = False
            print(
                "finishing up {} labels for {}...".format(labels_type, self._split_name)
            )

            for _ in label_worker:
                pass

    @staticmethod
    def _create_label(info: Tuple[str, str, Dict[str, float], str]):
        image_id, label_type, labels, images_dir = info
        image_path = os.path.join(images_dir, "{}.jpg".format(image_id))
        label_path = os.path.join(
            images_dir, "{}.lab-{}.json".format(image_id, label_type)
        )

        try:
            if not os.path.exists(image_path):
                return  # do not create if image doesn't exist

            with open(label_path, "w") as label_file:
                json.dump(labels, label_file)
        except Exception as err:
            if label_path is not None and os.path.exists(label_path):
                try:
                    os.remove(label_path)
                except Exception:
                    pass
