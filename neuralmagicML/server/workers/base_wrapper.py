"""
Code related to wrappers for the job worker to handle running them
through the proper flow and update state to the manager and the dataabase.
"""

from typing import Dict, Any, Union, Callable
import threading
import time
import logging

from neuralmagicML.server.models import JobStatus, Job, database
from neuralmagicML.server.workers.base import BaseJobWorker

__all__ = ["JobCancelError", "JobWorkerWrapper"]


_LOGGER = logging.getLogger(__name__)


class JobCancelError(Exception):
    """
    Error raised if a job was canceled
    """

    def __init__(self, *args: object):
        super().__init__(*args)


class JobWorkerWrapper(object):
    """
    The wrapper for a job worker to handle running an instance
    through the proper flow and update state to the manager and the database.

    :param worker: the worker instance to run
    """

    def __init__(self, worker: BaseJobWorker):
        self._worker = worker
        self._done_callback = None  # type: Union[Callable[[], None], None]
        self._lock = threading.Lock()

        self._started = False
        self._progress = None
        self._progress_time = None
        self._completed = False
        self._canceling = False
        self._canceled = False
        self._errored = False
        self._error = None

    @property
    def job_id(self) -> str:
        """
        :return: the job id
        """
        return self._worker.job_id

    @property
    def worker(self) -> BaseJobWorker:
        """
        :return: the worker instance to run
        """
        return self._worker

    @property
    def started(self) -> bool:
        """
        :return: True if start has been called, False otherwise
        """
        with self._lock:
            return self._started

    @property
    def progress(self) -> Union[None, Dict[str, Any]]:
        """
        :return: current progress, if any, for the running job worker
        """
        with self._lock:
            return self._progress

    @property
    def completed(self) -> bool:
        """
        :return: True if the job is completed, False otherwise
        """
        with self._lock:
            return self._completed

    @property
    def canceling(self) -> bool:
        """
        :return: True if the job is being canceled, False otherwise
        """
        with self._lock:
            return self._canceling

    @property
    def canceled(self) -> bool:
        """
        :return: True if the job is canceled, False otherwise
        """
        with self._lock:
            return self._canceled

    @property
    def errored(self) -> bool:
        """
        :return: True if the job has errored, False otherwise
        """
        with self._lock:
            return self._errored

    @property
    def error(self) -> Union[str, None]:
        """
        :return: The error, if any, encountered while running the job worker
        """
        with self._lock:
            return self._error

    def start(self, done_callback: Callable[[], None]):
        """
        Start running the contained job worker in a separate thread

        :param done_callback: the callback to invoke once completed running
        """
        _LOGGER.info(
            "starting job worker for job_id {} and project_id {}".format(
                self._worker.job_id, self._worker.project_id
            )
        )
        assert done_callback is not None

        with self._lock:
            if self._started:
                raise RuntimeError("start can only be called once")

            self._started = True
            self._done_callback = done_callback
            worker_thread = threading.Thread(target=self._worker_thread)
            worker_thread.start()

        _LOGGER.debug(
            "started job worker for job_id {} and project_id {}".format(
                self._worker.job_id, self._worker.project_id
            )
        )

    def cancel(self):
        """
        Cancel the running job. start must have been called first
        """
        _LOGGER.info(
            "canceling job worker for job_id {} and project_id {}".format(
                self._worker.job_id, self._worker.project_id
            )
        )

        with self._lock:
            if self._completed:
                return

            self._canceling = True

        # freeze the caller thread until canceled, completed, or error
        freeze = True

        while freeze:
            # don't hammer the CPU with constant checks
            time.sleep(0.01)

            with self._lock:
                freeze = not (self._errored or self._canceled or self._completed)

        _LOGGER.debug(
            "canceled job worker for job_id {} and project_id {}".format(
                self._worker.job_id, self._worker.project_id
            )
        )

    def _worker_thread(self):
        _LOGGER.debug(
            "job worker for job_id {} and project_id {} thead init".format(
                self._worker.job_id, self._worker.project_id
            )
        )

        with database.connection_context():
            with self._lock:
                job = Job.get(Job.job_id == self._worker.job_id)
                self._report_started(job)

            canceled = False
            error = None

            try:
                # initial check to see if job was canceled before it started
                if self._should_cancel():
                    raise JobCancelError()

                for progress in self._worker.run():
                    with self._lock:
                        if self._should_report_progress():
                            self._report_progress(job, progress)

                        if self._should_cancel():
                            raise JobCancelError()
            except JobCancelError:
                canceled = True

                _LOGGER.debug(
                    "cancel job worker for job_id {} and project_id {} received".format(
                        self._worker.job_id, self._worker.project_id
                    )
                )
            except Exception as err:
                _LOGGER.info(
                    (
                        "job worker for job_id {} and project_id {} "
                        "encountered error: {}"
                    ).format(self._worker.job_id, self._worker.project_id, err)
                )
                error = err

            with self._lock:
                self._start_completed = True

                if canceled:
                    self._report_canceled(job)
                    _LOGGER.info(
                        "canceled job worker for job_id {} and project_id {}".format(
                            self._worker.job_id, self._worker.project_id
                        )
                    )
                elif error is not None:
                    self._report_error(job, str(error))
                    _LOGGER.info(
                        "errored job worker for job_id {} and project_id {}".format(
                            self._worker.job_id, self._worker.project_id
                        )
                    )
                else:
                    self._report_completed(job)
                    _LOGGER.info(
                        "completed job worker for job_id {} and project_id {}".format(
                            self._worker.job_id, self._worker.project_id
                        )
                    )

        self._done_callback()

    def _report_started(self, job: Job):
        self._started = True
        job.status = JobStatus.started
        job.save()

    def _should_report_progress(self) -> bool:
        # let's not hammer the database, limit progress saves to 10 per second
        return self._progress_time is None or time.time() - self._progress_time >= 0.1

    def _report_progress(self, job: Job, progress: Dict[str, Any]):
        self._progress = progress
        self._progress_time = time.time()
        job.progress = progress
        job.save()

    def _report_completed(self, job: Job):
        self._completed = True
        job.status = JobStatus.completed
        job.progress = None
        job.save()

    def _should_cancel(self) -> bool:
        if self._canceling:
            return True

        return not threading.main_thread().is_alive()

    def _report_canceled(self, job: Job):
        self._canceled = True
        job.status = JobStatus.canceled
        job.progress = None
        job.save()

    def _report_error(self, job: Job, error: str):
        self._errored = True
        self._error = error
        job.status = JobStatus.error
        job.error = error
        job.progress = None
        job.save()
