"""
Code related to efficiently downloading multiple files with parallel workers
"""

from typing import List, Tuple, Iterator, Callable
from urllib.request import urlopen
import tempfile
import shutil
import os
import multiprocessing

from .worker import ParallelWorker


__all__ = ["DownloadResult", "MultiDownloader"]


class DownloadResult(object):
    """
    A file result from a download
    """

    def __init__(self, id_: str, source: str, dest: str):
        """
        :param id_: unique id for the file
        :param source: source url the file was downloaded from
        :param dest: destination path the file was downloaded to
        """
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
        :param source_dests: A list of tuples containing info for downloading the files, tuple is expected to be
                             of the form: (unique_id, source url, destination path)
        :param downloaded_callback: a callback function to be called after a download has happened in a worker
                                    for any additional work needed before the file is completed
        :param num_workers: number of workers to download files, if < 1 scales to 2x the core count for the machine
        :param overwrite_files: True to overwrite previous files in the destination, False otherwise
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
            for _ in MultiDownloader.download(
                res.source, res.dest, self._overwrite_files, self._num_retries
            ):
                pass

            res.downloaded = True
        except _PreviouslyDownloadedError:
            res.downloaded = False
        except Exception as err:
            res.err = err

        if self._download_callback is not None:
            self._download_callback(res)

        return res

    @staticmethod
    def download(
        url_path: str, dest_path: str, overwrite: bool = False, num_retries: int = 3
    ) -> Iterator[float]:
        for _ in range(num_retries):
            try:
                for val in MultiDownloader._download_helper(
                    url_path, dest_path, overwrite
                ):
                    yield val
            except _PreviouslyDownloadedError as err:
                raise err
            except Exception:
                continue

            return

    @staticmethod
    def _download_helper(
        url_path: str, dest_path: str, overwrite: bool
    ) -> Iterator[float]:
        if not overwrite and os.path.exists(dest_path):
            raise _PreviouslyDownloadedError()

        with urlopen(url_path) as connection:
            meta = connection.info()
            content_length = (
                meta.getheaders("Content-Length")
                if hasattr(meta, "getheaders")
                else meta.get_all("Content-Length")
            )
            file_size = (
                int(content_length[0])
                if content_length is not None and len(content_length) > 0
                else None
            )
            downloaded_size = 0.0
            temp = tempfile.NamedTemporaryFile(delete=False)

            try:
                while True:
                    buffer = connection.read(8192)
                    if len(buffer) == 0:
                        break
                    temp.write(buffer)
                    downloaded_size += len(buffer)

                    if file_size is not None and file_size > 0:
                        yield float(downloaded_size) / float(file_size)

                temp.close()

                if os.path.exists(dest_path):
                    os.remove(dest_path)

                shutil.move(temp.name, dest_path)
            finally:
                temp.close()

                if os.path.exists(temp.name):
                    os.remove(temp.name)


class _PreviouslyDownloadedError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
