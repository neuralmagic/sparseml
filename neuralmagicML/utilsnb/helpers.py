"""
Helper functions for jupyter notebooks
"""

__all__ = [
    "check_pytorch_notebook_setup",
    "check_tensorflow_notebook_setup",
    "format_html",
]


def check_pytorch_notebook_setup():
    try:
        import torch

        if torch.__version__[0] != "1":
            raise Exception

        import torchvision

        import tensorboard
    except Exception:
        raise ModuleNotFoundError(
            "please install all requirements for neuralmagicML before continuing, "
            "these are listed in the requirements.txt folder and can be installed "
            "using the setup.py or pip -r ./requirements.txt. Additionally, "
            "torch>=1.0.0 and a compatable version of torchvision are required to "
            "use neuralmagicML.pytorch."
        )

    print("notebook setup check complete!")


def check_tensorflow_notebook_setup():
    try:
        import tensorflow

        version = [int(v) for v in tensorflow.__version__.split(".")]
        if version[0] != 1 or version[1] < 8:
            raise Exception

        from neuralmagicML.tensorflow.utils import tf_compat

        tf_compat.logging.set_verbosity(tf_compat.logging.ERROR)

        import tensorboard
    except Exception:
        raise ModuleNotFoundError(
            "please install all requirements for neuralmagicML before continuing, "
            "these are listed in the requirements.txt folder and can be installed "
            "using the setup.py or pip -r ./requirements.txt. Additionally, "
            "tensorflow>=1.8,<2.0 is required to use neuralmagicML.tensorflow."
        )

    print("notebook setup check complete!")


def format_html(
    message: str, header: str = None, color: str = None, italic: bool = False
):
    """
    Create a message formatted as html using the given parameters.
    Expected to be used in ipywidgets.HTML.

    :param message: the message string to display
    :param header: the header type to use for the html (h1, h2, h3, h4, etc)
    :param color: the color to apply as a style (black, red, green, etc)
    :param italic: True to format the HTML as italic, False otherwise
    :return: the message formatted as html
    """
    if not message:
        message = ""

    message = "<i>{}</i>".format(message) if italic else message
    color = 'style="color: {}"'.format(color) if color else ""
    obj = "span" if not header else header
    html = "<{} {}>{}</{}>".format(obj, color, message, obj)

    return html
