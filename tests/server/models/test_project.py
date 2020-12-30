import pytest
import os
from tests.server.helper import database_fixture
from sparseml.server.models import database, Project


@pytest.mark.parametrize(
    "param_fixtures,expected_params",
    [
        (
            {},
            {
                "name": "",
                "description": "",
                "training_optimizer": None,
                "training_epochs": None,
                "training_lr_init": None,
                "training_lr_final": None,
            },
        ),
        (
            {
                "name": "test project",
                "description": "test description",
                "training_optimizer": "adam",
                "training_epochs": 20,
                "training_lr_init": 0.005,
                "training_lr_final": 0.0005,
            },
            {
                "name": "test project",
                "description": "test description",
                "training_optimizer": "adam",
                "training_epochs": 20,
                "training_lr_init": 0.005,
                "training_lr_final": 0.0005,
            },
        ),
    ],
)
def test_project(database_fixture, param_fixtures, expected_params):
    try:
        project = Project.create(**param_fixtures)

        # Test created fields match expected
        assert project.name == expected_params["name"]
        assert project.description == expected_params["description"]
        assert project.training_optimizer == expected_params["training_optimizer"]
        assert project.training_epochs == expected_params["training_epochs"]
        assert project.training_lr_init == expected_params["training_lr_init"]
        assert project.training_lr_final == expected_params["training_lr_final"]

        # Test if model is in database
        query = Project.get_or_none(Project.project_id == project.project_id)
        assert query == project

        # Test model methods
        with pytest.raises(FileNotFoundError):
            project.validate_filesystem()

        project.setup_filesystem()
        project.validate_filesystem()
        assert os.path.exists(project.dir_path)

        assert project.dir_size == 0
        with open(os.path.join(project.dir_path, "test_file"), "w+") as test_file:
            test_file.write("some test data")

        assert project.dir_size != 0

        if project and os.path.exists(project.dir_path):
            project.delete_filesystem()
        assert not os.path.exists(project.dir_path)

        # Test update
        last_mod = project.modified
        test_name = "new name"
        project.name = test_name
        project.save()
        assert project.name == test_name
        assert project.modified != last_mod

        # Test delete
        Project.delete().where(Project.project_id == project.project_id).execute()
        query = Project.get_or_none(Project.project_id == project.project_id)
        assert query is None
    finally:
        if project and os.path.exists(project.dir_path):
            project.delete_filesystem()

        if project:
            try:
                Project.delete().where(
                    Project.project_id == project.project_id
                ).execute()
            except Exception as e:
                pass

        assert not os.path.exists(project.dir_path)
