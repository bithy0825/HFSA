import copy
import torch.nn as nn

models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def make_model(model_spec, args=None, load_sd=False):
    if not isinstance(model_spec, dict):
        raise TypeError("model_spec must be a dictionary")
    if "name" not in model_spec:
        raise KeyError("model_spec must contain the 'name' key")
    if "args" not in model_spec:
        raise KeyError("model_spec must contain the 'args' key")

    model_name = model_spec["name"]
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not registered")

    if args is not None:
        model_args = copy.deepcopy(model_spec["args"])
        model_args.update(args)
    else:
        model_args = model_spec["args"]

    model = models[model_name](**model_args)

    if load_sd:
        state_dict = model_spec["sd"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key[len("_orig_mod.") :]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)

    return model
