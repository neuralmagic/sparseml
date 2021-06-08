# Research

This folder contains various research projects using SparseML.
They are not intended to be actively maintained and may require specific versions of SparseML and dependencies.
All subfolders contain a requirements file to guarantee reproducibility.
Implementations that are popular or show promising results will be productized into the SparseML `src` code and `integrations` folder.

To run any research projects, cd into the desired project's directory and install from the requirements.txt file using the following:

```bash
pip install -r requirements.txt
```

If any issues are encountered, first try starting from a new virtual environment and install the requirements:
```bash
virtualenv -p python3 venv
```

If there are continued issues, contact the author(s) indicated at the top of the README of each project's directory.

## Highlights

Coming soon!

## Project Structure

Each research project structure should abide by the following standards:

```
README.md - readme file
requirements.txt - python requirements file
*.py - files to run the implementation
```

### README.md

The readme file should follow the given template:

```markdown
# Title

Author: @github_username

Summary text

## Usage

A usage description that highlights how to use and run the research, including any command-line commands, notebooks, APIs, etc.

## Results

A results section that highlights anything notable from the research, including paper citation links.

### requirements.txt

Generated at the last working state for the repo by running the following command:

```bash
pip freeze > requirements.txt
```
