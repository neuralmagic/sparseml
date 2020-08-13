import numpy
from scipy.stats import entropy


__all__ = ["kl_divergence"]


def kl_divergence(
    predicted: numpy.ndarray,
    expected: numpy.ndarray,
    zero_point: float = 0.0,
    min_value: float = 1.0,
) -> float:
    """
    Calculate the kl_divergence (entropy) between two input arrays.

    Shifts all values such that the zero_point is at one.
    If a value is lower, then sets it equal to 1.

    :param predicted: the first array to compare with
    :param expected: the second array to compare with
    :param zero_point: the zero point that should be used to shift values above 1
    :param min_value: the minimum value that all values will be truncated to
        if they are below
    :return: the calculated KL divergence
    """
    if predicted.shape != expected.shape:
        raise ValueError(
            "predicted shape of {} must match expected shape of {}".format(
                predicted.shape, expected.shape
            )
        )

    # shift everything to have a min of 1 for the entropy / kl_divergence equation
    predicted = predicted.flatten() - zero_point + min_value
    expected = expected.flatten() - zero_point + min_value

    predicted[predicted < min_value] = min_value
    expected[expected < min_value] = min_value

    divergence = entropy(predicted, expected)

    return divergence
