from code_takedown.utils import setup_logger
from code_takedown.evaluators import get_evaluator
from tqdm import tqdm

import os
import json
import pdb

class EvaluateWinrate:
    def __init__(self, config):
        self.config = config

    def run(self):
        setup_logger(self.config)

        evaluator = get_evaluator(self.config)

        ################## EVALUATING WINRATE ##################
        evaluator.evaluate()
        ################## EVALUATING WINRATE ##################





        


       