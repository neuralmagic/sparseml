import argparse
import glob
import os
import re
import tarfile

UNWANTED = r"__pycache__|.git|.venv"
RINOH_ROOT = "docs/_build/rinoh"
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
        "setup.py",
        "requirements.txt",
    ]

    if os.path.exists(name) and exists_ok:
        mode = "a"
    else:
        mode = "x"

    with tarfile.TarFile(name=name, mode=mode) as package:
        add_to_package(package, folders)
        add_to_package(package, [RINOH_ROOT], old_root=RINOH_ROOT, new_root="docs")
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

    parser.add_argument(
        "--exists_ok",
        action="store_true",
        help="Adds to tar file if it already exists locally, otherwise throws error if it exists.",
    )

    args = parser.parse_args()
    create_package(name=args.tar_name, exists_ok=args.exists_ok)


if __name__ == "__main__":
    main()
