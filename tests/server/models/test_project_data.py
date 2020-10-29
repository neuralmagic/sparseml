import pytest
import os
from tests.server.helper import database_fixture
from neuralmagicML.server.models import database, Project, ProjectData


@pytest.mark.parametrize(
    "param_fixtures,expected_params",
    [
        (
            {},
            {"file": None, "source": "uploaded", "job": None},
        ),
    ],
)
def test_project_data(database_fixture, param_fixtures, expected_params):
    try:
        project = Project.create()
        project_data = ProjectData.create(project=project, **param_fixtures)

        # Test created fields match expected
        assert project_data.project_id == project.project_id

        # Test if model is in database
        query = ProjectData.get_or_none(ProjectData.data_id == project_data.data_id)
        assert query == project_data

        # Test model methods
        with pytest.raises(FileNotFoundError):
            project_data.validate_filesystem()

        project_data.setup_filesystem()
        project_data.validate_filesystem()
        assert os.path.exists(project_data.dir_path)

        # Test update
        project_data.file = "test.npz"
        assert project_data.file == "test.npz"

        # Test data file
        with open(
            os.path.join(project_data.dir_path, project_data.file), "w+"
        ) as test_file:
            test_file.write("some test data")

        assert os.path.exists(project_data.file_path)
        project_data.validate_filesystem()

        project_data.delete_filesystem()
        assert not os.path.exists(project_data.file_path)

        # Test delete
        ProjectData.delete().where(
            ProjectData.data_id == project_data.data_id
        ).execute()
        query = ProjectData.get_or_none(ProjectData.data_id == project_data.data_id)
        assert query is None

    finally:
        if project and os.path.exists(project.dir_path):
            project.delete_filesystem()

        if project_data:
            try:
                ProjectData.delete().where(
                    ProjectData.data_id == project_data.data_id
                ).execute()
            except Exception as e:
                pass

        if project:
            try:
                Project.delete().where(
                    Project.project_id == project.project_id
                ).execute()
            except Exception as e:
                pass
        assert not os.path.exists(project.dir_path)
