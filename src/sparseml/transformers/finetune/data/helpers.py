from datasets import Dataset, load_dataset

def get_raw_dataset(data_args, cache_dir: str, **kwargs) -> Dataset:
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=cache_dir,
        **kwargs
    )

    return raw_datasets