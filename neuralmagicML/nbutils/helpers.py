__all__ = ["format_html"]


def format_html(
    message: str, header: str = None, color: str = None, italic: bool = False
):
    if not message:
        message = ""

    message = "<i>{}</i>".format(message) if italic else message
    color = 'style="color: {}"'.format(color) if color else ""
    obj = "span" if not header else header
    html = "<{} {}>{}</{}>".format(obj, color, message, obj)

    return html
