import pytest


from neuralmagicML.utilsnb import format_html


@pytest.mark.parametrize("message", ["test message one", "test message two"])
def test_format_html(message):
    html = format_html(message)
    assert message in html
