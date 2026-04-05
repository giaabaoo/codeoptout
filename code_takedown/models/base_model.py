import os
import json
import pdb
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
import timeit
from pathlib import Path
import copy
from .prompt_utils import apply_prompt_template


class BaseModel:
    '''
    Base model class for loading models from HunggingFace
    '''
    def __init__(self, config, initialize=True):
        self.config = config
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if initialize:
            self._initialize_model()

    def _initialize_model(self):
        login(token=self.config.model.hf_token)

        if self.config.pipeline.name == "evaluate_unlearning_takedown":
            self.load_unlearning_model()
        else:
            self.load_model()
        
        try:
            if self.config.method.intervention_name == "speculative_r_cad":
                self.load_assistant_model()
        except:
            pass
    
    def copy(self):
        new_instance = BaseModel(self.config, initialize=False)  # Set initialize to False to avoid logging in HF again
        new_instance.type = self.type
        new_instance.model = self.model  
        new_instance.tokenizer = self.tokenizer 
        new_instance.model.seqlen = self.model.seqlen
        return new_instance
    
    def process_logits(self):
        prior_processor = self.model._get_logits_processor

        def new_logits_processor(*args, **kwargs):
            prior = prior_processor(*args, **kwargs)
            prior.append(TopKPerturbationLogitsProcessor(self.tokenizer, self.model, self.config.method.std))
            return prior
        
        self.model._get_logits_processor = new_logits_processor

    def load_assistant_model(self):
        self.type = 'speculative_decoding'
        if self.config.method.assistant_setting == 'ft':
            full_ft_path = f"{self.config.model.ft_path}/{self.config.model.name}-finetune-{self.config.dataset.ft_data}"
            self.assistant_model = AutoModelForCausalLM.from_pretrained(
                full_ft_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        elif self.config.method.assistant_setting == 'base':
            self.assistant_model = AutoModelForCausalLM.from_pretrained(
                self.config.model.hf_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )

    def load_model(self):
        self.type = 'normal'
        if self.config.model.setting == 'ft':
            full_ft_path = f"{self.config.model.ft_path}/{self.config.model.name}-finetune-{self.config.dataset.ft_data}"
            self.model = AutoModelForCausalLM.from_pretrained(
                full_ft_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(full_ft_path)  
        elif self.config.model.setting == 'base':
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.hf_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.hf_path)  

        self.model.seqlen = self.model.config.max_position_embeddings
        
        try:
            assert 'r_cad' in self.config.method.intervention_name 
        except:
            # since we modify the transformer library, we need to set this to avoid errors for the non-cad methods
            self.model.generation_config.context_aware_decoding_alpha = None

    def load_unlearning_model(self):
        full_ft_path = f"{self.config.model.ft_path}/{self.config.model.name}-finetune-{self.config.dataset.ft_data}/{self.config.method.unlearn_model_path}"

        self.model = AutoModelForCausalLM.from_pretrained(
            full_ft_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(full_ft_path) 
        self.type = 'unlearning'
        self.model.seqlen = self.model.config.max_position_embeddings

        try:
            assert 'r_cad' in self.config.method.intervention_name 
        except:
            # since we modify the transformer library, we need to set this to avoid errors for the non-cad methods
            self.model.generation_config.context_aware_decoding_alpha = None

    
    def inference(self, task_id, prompt, gt, task):
        if self.config.model.setting == 'ft' or self.config.model.setting == 'base':
            context = ''
        elif self.config.model.setting == 'rag':
            context = f"Copyrighted code: {gt}\n" + 'Please complete the following code: \n' + prompt
        
        
        template_prompt = apply_prompt_template(prompt_template_style='code', dataset=[prompt], context=context, model='code')[0]
        inputs = self.tokenizer(template_prompt, return_tensors="pt").to(self.model.device)
        
        # pdb.set_trace()
        
        time_start = timeit.default_timer()
        generate_ids = self.model.generate(self.tokenizer, inputs.input_ids, max_new_tokens=self.config.model.completion_len, do_sample=False, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id, attention_mask=inputs.attention_mask) # None but greedy decoding
        time_end = timeit.default_timer()

        inference_time = time_end - time_start

        outputs = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)  
        final_outputs, full_generated_outputs = self.postprocess(prompt, template_prompt, task, outputs)
        
        self.save_solution(task_id, full_generated_outputs, gt, task)

        return final_outputs, full_generated_outputs, inference_time

    def postprocess(self, prompt, template_prompt, task, outputs):
        if 'gsm' in str(task):
            outputs = [task._stop_at_stop_token(o.replace(template_prompt, '\n'), task.stop_words) for o in outputs]
        
            outputs = outputs[0]

            start_index = outputs.find('def')
            outputs = outputs[start_index:]

            if "assert" in outputs:
                stop_index = outputs.find("\"\"\"")
                outputs = outputs[:stop_index]

        else:
            outputs = [task._stop_at_stop_token(o.replace(template_prompt, '\n'), task.stop_words) for o in outputs]
            outputs = outputs[0]
           
        if not outputs:
            print(outputs)
            outputs = "return"

        full_generated_outputs = [prompt + "\n" + outputs]

        return outputs, full_generated_outputs

    def save_solution(self, task_id, full_generated_outputs, gt, task):
        # Write solution to file
        solution_path = f"{self.config.output_dir}/output_solutions"
        Path(solution_path).mkdir(parents=True, exist_ok=True)
        if 'humaneval' in str(task):
            task_id = int(task_id.replace("HumanEval/", ""))
            file_name = f"humaneval_problem_{task_id}"
        elif 'mbpp' in str(task):
            file_name = f"mbpp_problem_{task_id}"
        elif 'gsm' in str(task):
            file_name = f"gsm8k_problem_{task_id}"
            
        file_path = f"{self.config.output_dir}/output_solutions/{file_name}.py"

        response = "# SOLUTION: \n" + full_generated_outputs[0] + f"\n# GROUND_TRUTH: \n\'\'\'{gt}\n\'\'\'"  
        with open(file_path, 'w', encoding='utf-8') as file: file.write(response)

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
        order = torch.argsort(-scores, 1) # "-"" mean descending order
        order = order.cpu().detach().numpy()
        order_top_k = order[:, :top_k]
        batch_size = input_ids.shape[0]
        
        for ex in range(batch_size):
            # add gaussian noise to the top_k scores
            noise = torch.normal(0, self.std, size=(top_k,), device=scores.device)
            scores[ex][order_top_k[ex]] += noise
        return scores