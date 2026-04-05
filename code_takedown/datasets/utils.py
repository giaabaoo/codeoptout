from code_takedown.utils import Config
from .code_dataset import CodeDataset
from .code_split_dataset import CodeSplitDataset

def get_data_loader(config: Config):
    data_loaders = {
        'code': CodeDataset,
        'code_split': CodeSplitDataset
    }

    return data_loaders[config.dataset.dataset_class](config)
