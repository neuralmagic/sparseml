from setuptools import setup, find_packages


packages = find_packages(
    include=["neuralmagicML", "neuralmagicML.*"], exclude=["*.__pycache__.*"]
)

with open("requirements.txt", "r") as req_file:
    install_reqs = []

    for line in req_file.readlines():
        if line.strip() and not line.startswith("#"):
            install_reqs.append(line)

setup(
    name="neuralmagicML",
    version="1.1.0",
    packages=packages,
    package_data={},
    install_requires=install_reqs,
    author="Neural Magic",
    author_email="support@neuralmagic.com",
    url="https://neuralmagic.com/",
    license_file="LICENSE",
)
