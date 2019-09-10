from typing import Tuple, Dict, Callable
from torch.utils.data import Dataset


__all__ = ['create_dataset']


DATASET_MAPPINGS = {}  # type: Dict[str, Tuple[Callable, int]]


def create_dataset(dataset_type: str, root: str, train: bool, rand_trans: bool, **kwargs) -> Tuple[Dataset, int]:
    if dataset_type == 'help':
        raise Exception('dataset_type given of help, available datasets: \n{}'.format(list(DATASET_MAPPINGS.keys())))

    if dataset_type not in DATASET_MAPPINGS:
        raise ValueError('Unsupported dataset_type given of {}, available datasets: \n{}'
                         .format(dataset_type, list(DATASET_MAPPINGS.keys())))

    if dataset_type not in DATASET_MAPPINGS:
        raise ValueError('Unsupported dataset_type given of {}'.format(dataset_type))

    constructor, num_classes = DATASET_MAPPINGS[dataset_type]
    dataset = constructor(root=root, train=train, rand_trans=rand_trans, **kwargs)

    return dataset, num_classes
