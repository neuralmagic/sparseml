import os
import json
import shutil
import torch
from torchvision import transforms
from torchvision.datasets import Kinetics400

import cv2
from pytube import YouTube

import urllib.request as request
import tarfile

from neuralmagicML.pytorch.datasets.registry import DatasetRegistry
from neuralmagicML.pytorch.datasets.generic import default_dataset_path

__all__ = ["Kinetics400Dataset"]

_RGB_MEANS = [0.43216, 0.394666, 0.37645]
_RGB_STDS = [0.22803, 0.22145, 0.216989]


KINETICS_400_URL = (
    "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz"
)


def transform_video_tensor(tensor, transform):
    """
    Applies transform on video tensors, which are shaped

    :param tensor: Tensor[T, H, W, C] where T is the video frame
    :param transform: A function/transform that  takes in a CxHxW image
    :return: A Tensor[C, T, H, W]
    """
    new_tensor = tensor.transpose(1, 3)
    new_tensor = new_tensor.transpose(2, 3)
    tensor_split = list(torch.chunk(new_tensor, 30, dim=0))
    for i in range(len(tensor_split)):
        sub_tensor = tensor_split[i]
        sub_tensor = sub_tensor.reshape(sub_tensor.shape[1:])
        sub_tensor = transform(sub_tensor)
        tensor_split[i] = sub_tensor

    new_tensor = torch.stack(tensor_split)
    new_tensor = new_tensor.transpose(0, 1)
    return new_tensor


def convert_video(source, target, start_time, end_time, frames_per_clip):
    """
    :param source: the mp4 video location
    :param target: target directory where kinetics dataset will be store
    :param start_time: start frame of clip used in kinetics
    :param end_time: end frame of clip used in kinetics
    :param frames_per_clip: the framerate used to filter video
    """
    video_cap = cv2.VideoCapture(source)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    if round(fps) != frames_per_clip:
        return False
    curr_frame = 0
    start_time = round(fps * start_time)
    end_time = round(fps * end_time)
    success, frame = video_cap.read()

    height, width, _ = frame.shape
    video_writer = cv2.VideoWriter(
        target, cv2.VideoWriter_fourcc("M", "J", "P", "G"), fps, (width, height)
    )
    while success:
        if curr_frame >= start_time and curr_frame < end_time:
            video_writer.write(frame)
        elif curr_frame >= end_time:
            break
        curr_frame += 1
        success, frame = video_cap.read()
    return True


def scrape_videos(source, target, frames_per_clip, total_clips=10):
    """
    Scrape videos from a kinetics json db

    :param source: json file containing kinetics video information
    :param target: target directory where kinetics dataset will be store
    :param frames_per_clip: the framerate used to filter video
    :param total_clips: total number of clips that the database will contain
    """
    os.makedirs(target, exist_ok=True)
    with open(source) as file_data:
        video_data = json.load(file_data)
    total = 0
    for key in video_data.keys():
        if total > total_clips:
            return
        try:
            video_url = video_data[key]["url"]
            start_time, end_time = video_data[key]["annotations"]["segment"]
            label = video_data[key]["annotations"]["label"].replace(" ", "_")
            video = YouTube(video_url).streams.first()
            mp4_file = f"{target}_tmp/{total:04d}.mp4"
            os.makedirs(f"{target}/{label}", exist_ok=True)
            avi_file = f"{target}/{label}/{total:04d}.avi"
            video.download(output_path=f"{target}_tmp/", filename=f"{total:04d}")
            if convert_video(mp4_file, avi_file, start_time, end_time, frames_per_clip):
                total += 1
        except KeyboardInterrupt as e:
            raise e
        except:
            continue

    shutil.rmtree(f"{target}_tmp")


def download_kinetics(
    source_directory, target_directory, train, frames_per_clip, total_clips
):
    """
    Download kinetics database

    :param source_directory: source directory where tar file will be downloaded
    :param target_directory: target directory where kinetics dataset will be store
    :param train: true if downloading from training set, false if downloading from validation set
    :param frames_per_clip: the framerate used to filter video
    :param total_clips: total number of clips that the database will contain
    """
    os.makedirs(source_directory, exist_ok=True)

    kinetics_tar_path = os.path.join(source_directory, "kinetics400.tar.gz")
    request.urlretrieve(KINETICS_400_URL, kinetics_tar_path)
    with tarfile.open(kinetics_tar_path, "r") as tar_file:
        tar_file.extractall(source_directory)

    json_file = "train.json" if train else "validate.json"
    json_file = f"{source_directory}/kinetics400/{json_file}"
    scrape_videos(
        json_file,
        target_directory,
        frames_per_clip=frames_per_clip,
        total_clips=total_clips,
    )


@DatasetRegistry.register(
    key=["kinetics400", "kinetics_400"],
    attributes={
        "num_classes": 21,
        "transform_means": _RGB_MEANS,
        "transform_stds": _RGB_STDS,
    },
)
class Kinetics400Dataset(Kinetics400):
    """
    Wrapper for the Kinetics400 dataset to apply standard transforms.

    :param root: The root folder to find the dataset at, if not found will
        download here if download=True
    :param frames_per_clip: number of frames in a clip
    :param step_between_clips: number of frames between each clip
    :param train: True if this is for the training distribution,
        False for the validation
    :param rand_trans: True to apply RandomCrop and RandomHorizontalFlip to the data,
        False otherwise
    :param download: True to download the dataset, False otherwise.
    :param image_size: the size of the image to output from the dataset
    :param total_clips: the total number of clips autodownloaded
    """

    def __init__(
        self,
        root: str = default_dataset_path("kinetics-400"),
        frames_per_clip: int = 30,
        step_between_clips: int = 1,
        train: bool = True,
        rand_trans: bool = False,
        download: bool = True,
        image_size: int = 112,
        total_clips: int = 200,
    ):
        root = os.path.abspath(os.path.expanduser(root))
        directory = os.path.join(root, "train" if train else "val")

        if not os.path.isdir(directory) and download:
            download_kinetics(
                root,
                directory,
                train,
                frames_per_clip=frames_per_clip,
                total_clips=total_clips,
            )
        elif not os.path.isdir(directory):
            raise Exception()

        trans = (
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
            ]
            if rand_trans
            else [transforms.ToPILImage(), transforms.Resize((image_size, image_size))]
        )
        trans.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=_RGB_MEANS, std=_RGB_STDS),
            ]
        )

        transform = transforms.Lambda(
            lambda x: transform_video_tensor(x, transforms.Compose(trans))
        )

        super().__init__(
            directory,
            frames_per_clip=frames_per_clip,
            step_between_clips=1,
            transform=transform,
        )
