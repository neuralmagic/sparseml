import glob
import logging
import os

import numpy as np


class Dataloader:
    def __init__(
        self, files_glob, batch_size=1, expected_shape=None, expected_type=None
    ):
        self.files = sorted(glob.glob(files_glob))
        self.index = 0
        self.batch_size = batch_size
        self.batches = int(len(self.files) / self.batch_size)

        self.expected_shape = expected_shape
        self.expected_type = expected_type

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        self.index = 0
        return self

    def verify_array(self, array):
        if self.expected_shape is None:
            self.expected_shape = [element.shape for element in array]
        if self.expected_type is None:
            self.expected_type = [element.dtype for element in array]

        if [element.shape for element in array] != self.expected_shape:
            raise Exception(
                f"Shape does not match expected shape {self.expected_shape}"
            )

        if [element.dtype for element in array] != self.expected_type:
            raise Exception(f"Type does not match expected type {self.expected_type}")

    def __next__(self):
        if self.index >= len(self.files):
            raise StopIteration

        data = None
        while data is None or data[0].shape[0] < self.batch_size:
            if self.index >= len(self.files):
                logging.debug("Dataloader ran out of files to read")
                raise StopIteration

            npz_array = np.load(self.files[self.index])
            array_element = [npz_array[key] for key in npz_array]
            try:
                self.verify_array(array_element)

                if data is None:
                    data = [np.array([array]) for array in array_element]
                else:
                    data = [
                        np.append(data[index], [array], axis=0)
                        for index, array in enumerate(array_element)
                    ]
            except Exception as e:
                logging.error(e)
            self.index += 1
        return data
