SUPPORTED_MODELS = ["opt", "mpt", "llama2"]


def load_model(args):
    key = _get_model_key(args)
    if key == "opt":
        from opt import load_model as _load_model
    elif key == "llama2":
        from llama2 import load_model as _load_model
    else:
        from mpt import load_model as _load_model

    return _load_model(args)


def load_data(args):
    key = _get_model_key(args)
    if key == "opt":
        from opt import load_data as _load_data
    elif key == "llama2":
        from llama2 import load_data as _load_data
    else:
        from mpt import load_data as _load_data

    return _load_data(args)


def prepare_sparsegpt(model, dataloader, args, **kwargs):
    key = _get_model_key(args)
    if key == "opt":
        from opt import prepare_sparsegpt as _prepare_sparsegpt
    else:
        from mpt import prepare_sparsegpt as _prepare_sparsegpt

    return _prepare_sparsegpt(model, dataloader, args, **kwargs)


def _get_model_key(args):
    key = None
    for k in SUPPORTED_MODELS:
        if args.model.lower().find(k) >= 0:
            key = k
            break
    if key is None:
        raise ValueError(
            f"Model {args.model} is not supported. Supported models: {SUPPORTED_MODELS.keys()}"
        )
    return key
