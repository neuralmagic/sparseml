import tensorflow as tf


__all__ = ["get_layer_name_from_param"]


def get_layer_name_from_param(param: str):
    known_weights = ["kernel", "bias"]
    pos = param.rfind("/")
    if pos > -1:
        suff = param[pos + 1 :]
        found = False
        for s in known_weights:
            colon_pos = suff.rfind(":")
            if suff[:colon_pos] == s:
                found = True
                break
        if not found:
            raise ValueError(
                "Unrecognized weight names. Expected: ".format(known_weights)
            )
    return param[:pos]
