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
General code for parallelizing the workers
"""

import time
from queue import Empty, Full, Queue
from threading import Thread
from typing import Any, Callable, Iterator, List


__all__ = ["ParallelWorker"]


class ParallelWorker(object):
    """
    Multi threading worker to parallelize tasks

    :param worker_func: the function to parallelize across multiple tasks
    :param num_workers: number of workers to use
    :param indefinite: True to keep the thread pooling running so that
        more tasks can be added, False to stop after no more tasks are added
    :param max_source_size: the maximum size for the source queue
    """

    def __init__(
        self,
        worker_func: Callable,
        num_workers: int,
        indefinite: bool,
        max_source_size: int = -1,
    ):
        self._worker_func = worker_func
        self._num_workers = num_workers

        self._pending_count = 0
        self._source_queue = (
            Queue(maxsize=max_source_size) if max_source_size > 0 else Queue()
        )
        self._completed = Queue()
        self._indefinite = Queue()
        self._shutdown = Queue()

        self.indefinite = indefinite

    def __iter__(self) -> Iterator[Any]:
        while self._shutdown.empty() and not (
            self._indefinite.empty()
            and self._pending_count < 1
            and self._completed.empty()
        ):
            try:
                res = self._completed.get(block=True, timeout=1.0)
                self._pending_count -= 1

                yield res
            except Empty:
                continue

    def __len__(self):
        return self._pending_count

    @property
    def indefinite(self) -> bool:
        """
        :return: True to keep the thread pooling running so that
            more tasks can be added, False to stop after no more tasks are added
        """
        return not self._indefinite.empty()

    @indefinite.setter
    def indefinite(self, value: bool):
        """
        :param value: True to keep the thread pooling running so that
            more tasks can be added, False to stop after no more tasks are added
        """
        if value and self._indefinite.empty():
            self._indefinite.put(True)
        elif not value and not self._indefinite.empty():
            self._indefinite.get()

    def start(self):
        """
        Start the workers
        """
        for _ in range(self._num_workers):
            Thread(
                target=ParallelWorker._worker,
                args=(
                    self._worker_func,
                    self._source_queue,
                    self._completed,
                    self._indefinite,
                    self._shutdown,
                ),
            ).start()

    def shutdown(self):
        """
        Stop the workers
        """
        self._shutdown.put(True)

    def add(self, vals: List[Any]):
        """
        :param vals: the values to add for processing work
        """
        self._pending_count += len(vals)
        ParallelWorker._adder(vals, self._source_queue, self._shutdown)

    def add_async(self, vals: List[Any]):
        """
        :param vals: the values to add for async workers
        """
        self._pending_count += len(vals)
        Thread(
            target=ParallelWorker._adder,
            args=(vals, self._source_queue, self._shutdown),
        ).start()

    def add_async_generator(self, gen: Iterator[Any]):
        """
        :param gen: add an async generator to pull values from for processing
        """
        Thread(
            target=ParallelWorker._gen_adder,
            args=(gen, self._source_queue, self._shutdown, self._indefinite),
        ).start()

    def add_item(self, val: Any):
        """
        :param val: add a single item for processing
        """
        self._pending_count += 1
        self._source_queue.put(val)

    @staticmethod
    def _worker(
        worker_func: Callable,
        source_queue: Queue,
        completed: Queue,
        indefinite: Queue,
        shutdown: Queue,
    ):
        while True:
            if not shutdown.empty() or (source_queue.empty() and indefinite.empty()):
                return

            try:
                val = source_queue.get(block=True, timeout=0.01)
            except Empty:
                continue

            res = worker_func(val)
            completed.put(res)
            source_queue.task_done()

    @staticmethod
    def _adder(vals: List[Any], source_queue: Queue, shutdown: Queue):
        index = 0

        while index < len(vals) and shutdown.empty():
            try:
                source_queue.put(vals[index], block=True, timeout=0.01)
                index += 1
            except Full:
                continue

    @staticmethod
    def _gen_adder(
        gen: Iterator[Any], source_queue: Queue, shutdown: Queue, indefinite: Queue
    ):
        indefinite.put(True)

        for val in gen:
            while True:
                if not shutdown.empty():
                    return

                try:
                    source_queue.put(val, block=True, timeout=0.01)
                    break
                except Full:
                    continue

        # give some time for everything to complete since we didn't add to pending count
        # need to architect this better in the future to get rid of
        # the edge case (last items don't complete in 1 sec)
        while not source_queue.empty():
            time.sleep(0.1)

        time.sleep(1.0)

        while not indefinite.empty():
            try:
                indefinite.get_nowait()
            except Empty:
                continue
