from code_takedown.utils import Config
from .general_takedown_method import GeneralTakedownMethod
from .ke_takedown_method import KETakedownMethod
from .decoding_takedown_method import DecodingTakedownMethod

def get_takedown_method(config: Config):
    takedown_methods = {
        'general': GeneralTakedownMethod,
        'training_based_on_single_sample': KETakedownMethod,
        'decoding': DecodingTakedownMethod
    }

    return takedown_methods[config.method.name](config)