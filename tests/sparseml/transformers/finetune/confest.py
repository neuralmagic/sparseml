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

import csv
import json
import os
from io import StringIO

import pytest


# @pytest.fixture(scope="function")
# def create_mock_files_fixture(request):
#     def create_mock_file(extension, content, path, filename):
#         os.makedirs(path, exist_ok=True)

#         if extension == "json":
#             # Create mock JSON data
#             mock_data = {"text": content}
#             # Serialize the mock data to a JSON string
#             mock_content = json.dumps(mock_data, indent=2)

#         elif extension == "csv":
#             # Create mock CSV data
#             fieldnames = ["text"]
#             mock_data = [{"text": content}]
#             # Serialize the mock data to a CSV string
#             csv_output = StringIO()
#             csv_writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
#             csv_writer.writeheader()
#             csv_writer.writerows(mock_data)
#             mock_content = csv_output.getvalue()

#         else:
#             raise ValueError("Unsupported file extension")

#         # Determine the file extension based on the fixture parameter
#         file_extension = "json" if extension == "json" else "csv"

#         # Create a mock file with a unique name
#         mock_filename = f"{filename}.{file_extension}"
#         mock_filepath = os.path.join(path, mock_filename)

#         # Write the mock content to the file
#         with open(mock_filepath, "w") as mock_file:
#             mock_file.write(mock_content)

#         return mock_filepath  # Return the file path

#     yield create_mock_file  # Yield the inner function


@pytest.fixture
def create_mock_files_fixture(extension, content, path, filename):
    os.makedirs(path, exist_ok=True)

    if extension == "json":
        # Create mock JSON data
        mock_data = {"text": content}
        # Serialize the mock data to a JSON string
        mock_content = json.dumps(mock_data, indent=2)

    elif extension == "csv":
        # Create mock CSV data
        fieldnames = ["text"]
        mock_data = [{"text": content}]
        # Serialize the mock data to a CSV string
        csv_output = StringIO()
        csv_writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(mock_data)
        mock_content = csv_output.getvalue()

    else:
        raise ValueError("Unsupported file extension")

    # Determine the file extension based on the fixture parameter
    file_extension = "json" if extension == "json" else "csv"

    # Create a mock file with a unique name
    mock_filename = f"{filename}.{file_extension}"
    mock_filepath = os.path.join(path, mock_filename)

    # Write the mock content to the file
    with open(mock_filepath, "w") as mock_file:
        mock_file.write(mock_content)

    yield mock_filepath  # Return the file path
