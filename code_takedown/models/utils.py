from code_takedown.utils import Config
from .base_model import BaseModel


def get_model(config: Config):
    models = {
        'base': BaseModel
    }

    return models[config.model.model_class](config)