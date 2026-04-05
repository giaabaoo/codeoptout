from code_takedown.utils import Config
from .code_evaluator import CodeEvaluator
from .winrate_evaluator import WinrateEvaluator
def get_evaluator(config: Config):
    evaluators = {
        'code': CodeEvaluator,
        'winrate': WinrateEvaluator
    }

    return evaluators[config.evaluator.evaluator_class](config)
