import sys
import os


def setup_path(notebook_nested_count: int = 0):
    if 'WORKBOOK_DIR' not in globals():
        globals()['WORKBOOK_DIR'] = os.path.expanduser(os.getcwd())

    # Adding the path to the parent directory so that neuralmagicML and notebookwidgets will be available
    # If this step does not work, then the path will need to be added manually
    package_path = os.path.abspath(os.path.join(globals()['WORKBOOK_DIR'],
                                                *[os.pardir for _ in range(notebook_nested_count + 1)]))
    helpers_path = os.path.abspath(os.path.join(globals()['WORKBOOK_DIR'],
                                                *[os.pardir for _ in range(notebook_nested_count)]))
    sys.path.extend([package_path, helpers_path])

    print('Python %s on %s' % (sys.version, sys.platform))
    print('Workbook dir: {}'.format(globals()['WORKBOOK_DIR']))
    print('Added current package path to sys.path: {}'.format(package_path))
    print('Added notebooks path to sys.path: {}'.format(helpers_path))
    print('Be sure to install from requirements.txt separately')
