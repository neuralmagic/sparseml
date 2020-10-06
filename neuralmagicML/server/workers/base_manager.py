"""
Code related to managing jobs in the server
"""

from typing import Union
import logging
import threading

from neuralmagicML.utils import Singleton
from neuralmagicML.server.models import database, Job, JobStatus
from neuralmagicML.server.workers.base_wrapper import JobWorkerWrapper
from neuralmagicML.server.workers.base import JobWorkerRegistryHolder


__all__ = ["JobNotFoundError", "JobCancelationFailureError", "JobWorkerManager"]


_LOGGER = logging.getLogger(__name__)


class JobNotFoundError(Exception):
    """
    Error raised if a job is not found in the database
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class JobCancelationFailureError(Exception):
    """
    Error raised if a job could not be canceled
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class JobWorkerManager(object, metaclass=Singleton):
    """
    Manager class for handling running job workers in the background.
    Only one job worker can run at once.
    Once one completes, the next oldest one marked as pending in the db is launched.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._current = None  # type: Union[None, JobWorkerWrapper]

    def app_startup(self):
        """
        Handle app startup to clear uncompleted state for jobs and begin running
        """

        # cancel any jobs that were left in an uncompleted state
        with database.connection_context():
            Job.update(status=JobStatus.canceled).where(Job.status == JobStatus.started)

        self.refresh()

    def refresh(self):
        """
        Refresh the available jobs.
        If a new job is marked as pending and no current job is running,
        will start the new job.

        Otherwise will exit out without doing anything and
        subsequent jobs will be launched after the current one completes.
        """
        refresh_thread = threading.Thread(target=self._refresh_worker)
        refresh_thread.start()

    def cancel_job(self, job_id: str):
        """
        Cancel a job with the given job_id so it won't be run.
        Blocks until the job can be canceled.

        :param job_id: the job_id to cancel
        :raise JobNotFoundError: if the job could not be found in the database
        :raise JobCancelationFailureError: if the job could not be canceled
        """
        _LOGGER.info("Canceling job with id {}".format(job_id))

        with self._lock:
            if self._current is not None and self._current.job_id == job_id:
                self._current.cancel()

                return

            with database.connection_context():
                job = Job.get_or_none(Job.job_id == job_id)

                if job is None:
                    _LOGGER.error("Could not find job with id {}".format(job_id))

                    raise JobNotFoundError(
                        "Could not find job with id {}".format(job_id)
                    )

                if (
                    job.status == JobStatus.error
                    or job.status == JobStatus.completed
                    or job.status == JobStatus.canceled
                ):
                    _LOGGER.error(
                        "Could not cancel job with status {}".format(job.status)
                    )

                    raise JobCancelationFailureError(
                        "Job with status {} cannot be canceled".format(job.status)
                    )

                job.status = JobStatus.canceled
                job.save()

    def _refresh_worker(self):
        _LOGGER.info("refreshing JobWorkerManager state")

        with self._lock:
            if (
                self._current is not None
                and not self._current.completed
                and not self._current.canceled
                and not self._current.errored
            ):
                return

            self._current = JobWorkerManager._load_next_pending()

            if self._current is not None:
                _LOGGER.info(
                    (
                        "found pending job with job_id {} "
                        "and project_id {}, starting"
                    ).format(
                        self._current.worker.job_id, self._current.worker.project_id
                    )
                )
                self._current.start(self.refresh)
            else:
                _LOGGER.info("no pending jobs found")

    @staticmethod
    def _load_next_pending() -> Union[None, JobWorkerWrapper]:
        _LOGGER.debug("loading next pending job for JobWorkerManager")
        err_count = 0

        while err_count < 5:
            try:
                worker = JobWorkerManager._load_next_pending_helper()

                return worker
            except Exception as err:
                _LOGGER.error(
                    (
                        "error while loading next pending job "
                        "for JobWorkerManager {}"
                    ).format(err)
                )
                err_count += 1

    @staticmethod
    def _load_next_pending_helper() -> Union[None, JobWorkerWrapper]:
        with database.connection_context():
            next_job = None  # type: Union[None, Job]
            query = (
                Job.select()
                .where(Job.status == JobStatus.pending)
                .order_by(Job.created)
                .limit(1)
            )

            for job in query:
                next_job = job
                break

            if next_job is None:
                return None

            try:
                if next_job.type_ not in JobWorkerRegistryHolder.REGISTRY:
                    raise ValueError(
                        "Cannot find job of type {}".format(next_job.type_)
                    )

                cls = JobWorkerRegistryHolder.REGISTRY[next_job.type_]
                worker = cls(job.job_id, job.project_id, **job.worker_args)
                wrapper = JobWorkerWrapper(worker)

                return wrapper
            except Exception as err:
                next_job.error = str(err)
                next_job.status = JobStatus.error
                next_job.save()

                raise err
