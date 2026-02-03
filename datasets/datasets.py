import copy
from torch.utils.data import Dataset

datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls

    return decorator


def make_dataset(dataset_spec, args=None):
    if not isinstance(dataset_spec, dict):
        raise TypeError("dataset_spec must be a dictionary")
    if "name" not in dataset_spec:
        raise KeyError("dataset_spec must contain the 'name' key")
    if "args" not in dataset_spec:
        raise KeyError("dataset_spec must contain the 'args' key")

    dataset_name = dataset_spec["name"]
    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' is not registered")

    if args is not None:
        dataset_args = copy.deepcopy(dataset_spec["args"])
        dataset_args.update(args)
    else:
        dataset_args = dataset_spec["args"]

    dataset = datasets[dataset_name](**dataset_args)

    return dataset
