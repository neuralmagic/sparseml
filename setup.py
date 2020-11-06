from setuptools import setup, find_packages
from sys import platform

packages = find_packages(
    include=["neuralmagicML", "neuralmagicML.*"], exclude=["*.__pycache__.*"]
)


def get_reqs(req_file_path):
    with open(req_file_path, "r") as req_file:

        install_reqs = []

        for line in req_file.readlines():
            if line.strip() and not line.startswith("#"):
                install_reqs.append(line)
        return install_reqs


install_reqs = get_reqs("requirements.txt")

if platform == "linux" or platform == "linux2":
    install_reqs += get_reqs("requirements-linux.txt")

setup(
    name="neuralmagicML",
    version="1.3.0",
    packages=packages,
    package_data={},
    include_package_data=True,
    install_requires=install_reqs,
    author="Neural Magic",
    author_email="support@neuralmagic.com",
    url="https://neuralmagic.com/",
    license_file="LICENSE",
    entry_points={
        "console_scripts": ["sparsify=neuralmagicML.server.app:main"],
    },
)
