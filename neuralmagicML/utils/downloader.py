"""
Code related to efficiently downloading multiple files with parallel workers
"""

from typing import List, Tuple, Iterator, Callable
import os
import multiprocessing
import requests
from tqdm import auto

from neuralmagicML.utils.worker import ParallelWorker
from neuralmagicML.utils.helpers import clean_path, create_parent_dirs


__all__ = [
    "PreviouslyDownloadedError",
    "download_file",
    "DownloadResult",
    "MultiDownloader",
]


class PreviouslyDownloadedError(Exception):
    """
    Error raised when a file has already been downloaded and overwrite is False
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def _download(
    url_path: str, dest_path: str, show_progress: bool, progress_title: str,
):
    request = requests.get(url_path, stream=True)
    request.raise_for_status()
    content_length = request.headers.get("content-length")

    try:
        content_length = int(content_length)
    except Exception:
        content_length = None

    progress = (
        auto.tqdm(
            total=content_length,
            desc=progress_title if progress_title else "downloading...",
        )
        if show_progress and content_length and content_length > 0
        else None
    )

    try:
        with open(dest_path, "wb") as file:
            for chunk in request.iter_content(chunk_size=1024):
                if not chunk:
                    continue

                file.write(chunk)
                file.flush()

                if progress:
                    progress.update(n=len(chunk))
    except Exception as err:
        os.remove(dest_path)
        raise err

    if progress:
        progress.close()


def download_file(
    url_path: str,
    dest_path: str,
    overwrite: bool,
    num_retries: int = 3,
    show_progress: bool = True,
    progress_title: str = None,
):
    """
    Download a file from the given url to the desired local path

    :param url_path: the source url to download the file from
    :param dest_path: the local file path to save the downloaded file to
    :param overwrite: True to overwrite any previous files if they exist,
        False to not overwrite and raise an error if a file exists
    :param num_retries: number of times to retry the download if it fails
    :param show_progress: True to show a progress bar for the download,
        False otherwise
    :param progress_title: The title to show with the progress bar
    :raise PreviouslyDownloadedError: raised if file already exists at dest_path
        nad overwrite is False
    """
    dest_path = clean_path(dest_path)
    create_parent_dirs(dest_path)

    if not overwrite and os.path.exists(dest_path):
        raise PreviouslyDownloadedError()

    if os.path.exists(dest_path):
        try:
            os.remove(dest_path)
        except OSError as err:
            print(
                "warning, error encountered when removing older "
                "cache_file at {}: {}".format(dest_path, err)
            )

    retry_err = None

    for _ in range(num_retries + 1):
        try:
            _download(url_path, dest_path, show_progress, progress_title)
            break
        except PreviouslyDownloadedError as err:
            raise err
        except Exception as err:
            retry_err = err

    if retry_err is not None:
        raise retry_err


class DownloadResult(object):
    """
    A file result from a download

    :param id_: unique id for the file
    :param source: source url the file was downloaded from
    :param dest: destination path the file was downloaded to
    """

    def __init__(self, id_: str, source: str, dest: str):
        self.id_ = id_
        self.source = source
        self.dest = dest
        self.err = None
        self.downloaded = False


class MultiDownloader(object):
    """
    Downloader to handle parallel download of multiple files at once
    """

    def __init__(
        self,
        source_dests: List[Tuple[str, str, str]],
        downloaded_callback: Callable[[DownloadResult], None] = None,
        num_workers: int = 0,
        overwrite_files: bool = False,
        num_retries: int = 3,
    ):
        """
        :param source_dests: A list of tuples containing info for downloading the files,
            tuple is expected to be of the form:
            (unique_id, source url, destination path)
        :param downloaded_callback: a callback function to be called after a download
            has happened in a worker for any additional work needed before the file is
            completed
        :param num_workers: number of workers to download files,
            if < 1 scales to 2x the core count for the machine
        :param overwrite_files: True to overwrite previous files in the destination,
            False otherwise
        :param num_retries: number of times to retry downloads for workers
        """
        if num_workers < 1:
            num_workers = round(
                2 * multiprocessing.cpu_count()
            )  # scale with the number of cores on the machine

        self._num_downloads = len(source_dests)

        if num_workers > self._num_downloads > 0:
            num_workers = self._num_downloads

        self._download_callback = downloaded_callback
        self._worker = ParallelWorker(self._worker_func, num_workers, indefinite=False)
        self._worker.add_async(source_dests)
        self._overwrite_files = overwrite_files
        self._num_retries = num_retries

    def __len__(self):
        return self._num_downloads

    def __iter__(self) -> Iterator[DownloadResult]:
        self._worker.start()

        for val in self._worker:
            yield val

    def _worker_func(self, val: Tuple[str, str, str]):
        res = DownloadResult(*val)

        try:
            download_file(
                res.source,
                res.dest,
                self._overwrite_files,
                self._num_retries,
                show_progress=False,
            )
        except PreviouslyDownloadedError:
            res.downloaded = False
        except Exception as err:
            res.err = err

        if self._download_callback is not None:
            self._download_callback(res)

        return res
