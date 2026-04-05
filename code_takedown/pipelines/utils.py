from code_takedown.utils import Config
from .get_forget_gt_vanilla import GetForgetGT
from .evaluate_vanilla import EvaluateVanilla
from .evaluate_takedown_at_inference import EvaluateTakedownAtInference
from .evaluate_takedown_at_decoding import EvaluateTakedownAtDecoding
from .evaluate_unlearning_takedown import EvaluateUnlearningTakedown
from .evaluate_takedown_at_training import EvaluateTakedownAtTraining
from .evaluate_winrate import EvaluateWinrate

def get_pipeline(config: Config):
    pipelines = {
        'get_forget_gt_vanilla': GetForgetGT,
        'evaluate_vanilla': EvaluateVanilla,
        'evaluate_takedown_at_inference': EvaluateTakedownAtInference,
        'evaluate_takedown_at_decoding': EvaluateTakedownAtDecoding,
        'evaluate_unlearning_takedown': EvaluateUnlearningTakedown,
        'evaluate_takedown_at_training': EvaluateTakedownAtTraining,
        'evaluate_winrate': EvaluateWinrate
    }

    return pipelines[config.pipeline.name](config)

