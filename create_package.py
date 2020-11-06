import argparse
import glob
import os
import re
import tarfile
import shutil

UNWANTED = r"__pycache__|.git|.venv"
DOC_BUILD_ROOT = "docs/build"
DIRECTORY_ROOT = "neuralmagicML-python"


def add_to_package(package, folders, old_root=None, new_root=""):
    while len(folders) > 0:
        folder = folders.pop(0)
        if os.path.isfile(folder):
            if old_root is not None:
                target_folder = folder.replace(old_root, new_root)
            else:
                target_folder = folder
            tarinfo = tarfile.TarInfo(os.path.join(DIRECTORY_ROOT, target_folder))
            package.add(folder, arcname=os.path.join(DIRECTORY_ROOT, target_folder))
            continue
        else:
            folder = os.path.join(folder, "*")
            for subfile in glob.glob(folder):
                if re.search(UNWANTED, subfile) is not None:
                    continue
                folders.append(subfile)


def create_package(name="neuralmagicML-python.tar.gz", exists_ok=True):
    folders = [
        "neuralmagicML",
        "notebooks",
        "scripts",
        "README.md",
        "MANIFEST.in",
        "setup.py",
        "requirements.txt",
        "requirements-linux.txt",
    ]

    if os.path.exists(name):
        shutil.rmtree(name)

    with tarfile.open(name=name, mode="x:gz") as package:
        add_to_package(package, folders)
        add_to_package(
            package, [DOC_BUILD_ROOT], old_root=DOC_BUILD_ROOT, new_root="docs"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Packages neuralmagicml into a tar file"
    )

    parser.add_argument(
        "--tar-name", type=str, required=True, help="Name of the tar file"
    )

    args = parser.parse_args()
    create_package(name=args.tar_name)


if __name__ == "__main__":
    main()
