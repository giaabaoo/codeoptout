import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import *
import numpy as np

class TopKPerturbationLogitsProcessor(LogitsProcessor):
    
    def __init__(self, tokenizer, model, std=0.1):
        self.tokenizer = tokenizer
        self.model = model
        self.std = std

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids = input_ids.cpu().detach().numpy()
        top_k = self.model.generation_config.top_k
        # Order the tokens by their likelihood
        order = torch.argsort(-scores, 1)
        order = order.cpu().detach().numpy()
        order_top_k = order[:, :top_k]
        batch_size = input_ids.shape[0]
        
        for ex in range(batch_size):
            # add gaussian noise to the top_k scores
            noise = torch.normal(0, self.std, size=(top_k,), device=scores.device)
            scores[ex][order_top_k[ex]] += noise
        return scores