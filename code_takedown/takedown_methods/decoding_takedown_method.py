import os
import json
import pdb
from .tool.prompt_utils import apply_prompt_template
from .tool.decoding_intervention import TopKPerturbationLogitsProcessor
import timeit
from transformers import GenerationConfig
from pathlib import Path
import torch
import ast
import inspect

class DecodingTakedownMethod:
    '''
    Base method class for taking down copyrights based on general strategy
    '''
    def __init__(self, config):
        self.config = config

    def takedown_speculative_r_cad(self, prompt, gt, tokenizer, model, assistant_model):
        if self.config.model.setting == 'ft':
            context = ''

        null_prompt = prompt + ' ' + gt        
        template_prompt = apply_prompt_template(prompt_template_style='code', dataset=[prompt], context=context, model='code')[0]

        inputs = tokenizer(template_prompt, return_tensors="pt").to(model.device)
        null_inputs = tokenizer(null_prompt, return_tensors="pt").to(model.device)
        generation_config = GenerationConfig(do_sample=False, num_return_sequences=1, context_aware_decoding_alpha=self.config.method.context_aware_decoding_alpha, max_new_tokens=self.config.model.completion_len)

        time_start = timeit.default_timer()
        with torch.no_grad():
            generate_ids = model.generate(tokenizer, assistant_model=assistant_model, inputs=inputs.input_ids, null_inputs=null_inputs.input_ids, attention_mask=inputs.attention_mask, generation_config=generation_config, pad_token_id=tokenizer.eos_token_id)
            
        time_end = timeit.default_timer()

        inference_time = time_end - time_start

        return generate_ids, inference_time, template_prompt

    def get_function_signature(self, func):
        # Parse the source code string into an AST
        tree = ast.parse(func)  # Parse the source code into an AST
        
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Return the function signature (e.g., 'def function_name(args):')
                return func.splitlines()[node.lineno - 1]  # Get the line containing the function definition
        
        return None  # Return None if no function is found

    def get_function_name(self, func):
        # Get the source code of the function
        # Parse the source code into an AST
        tree = ast.parse(func)
        
        # The function definition is the first node in the AST
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return node.name  # Return the function name
        
        return None  # In case the function is not found
    
    def takedown_FFT_r_cad(self, prompt, gt, tokenizer, model):
        if self.config.model.setting == 'ft':
            context = ''

        null_prompt = prompt + ' ' + gt        

        template_prompt = apply_prompt_template(prompt_template_style='code', dataset=[prompt], context=context, model='code')[0]

        inputs = tokenizer(template_prompt, return_tensors="pt").to(model.device)
        null_inputs = tokenizer(null_prompt, return_tensors="pt").to(model.device)
        
        time_start = timeit.default_timer()
        function_signature = self.get_function_signature(gt)

        if function_signature is None:
            pdb.set_trace()

        function_signature_tokens = tokenizer(function_signature, return_tensors="pt").input_ids[0]

        with torch.no_grad():
            # generation_config = GenerationConfig(do_sample=False, num_return_sequences=1, context_aware_decoding_alpha=None, max_new_tokens=len(function_signature_tokens))
            # generated_function_signature_ids = model.generate(tokenizer, inputs=inputs.input_ids, attention_mask=inputs.attention_mask,  generation_config=generation_config, pad_token_id=tokenizer.eos_token_id)
            # inputs.input_ids = generated_function_signature_ids
            # generated_function_name_attention_mask = torch.ones(generated_function_signature_ids.size(0), generated_function_signature_ids.size(1), dtype=inputs.attention_mask.dtype, device=inputs.attention_mask.device)
            # inputs.attention_mask = torch.cat([inputs.attention_mask, generated_function_name_attention_mask], dim=-1)
            
            generation_config = GenerationConfig(do_sample=False, num_return_sequences=1, context_aware_decoding_alpha=self.config.method.context_aware_decoding_alpha, max_new_tokens=self.config.method.num_first_tokens)
            first_few_tokens_ids = model.generate(tokenizer, inputs=inputs.input_ids, null_inputs=null_inputs.input_ids, attention_mask=inputs.attention_mask, generation_config=generation_config, pad_token_id=tokenizer.eos_token_id)
            inputs.input_ids = first_few_tokens_ids
            first_few_tokens_attention_mask = torch.ones(first_few_tokens_ids.size(0), first_few_tokens_ids.size(1), dtype=inputs.attention_mask.dtype, device=inputs.attention_mask.device)
            inputs.attention_mask = torch.cat([inputs.attention_mask, first_few_tokens_attention_mask], dim=-1)
            
            generation_config = GenerationConfig(do_sample=False, num_return_sequences=1, context_aware_decoding_alpha=None, max_new_tokens=self.config.model.completion_len)
            generate_ids = model.generate(tokenizer, inputs=inputs.input_ids, attention_mask=inputs.attention_mask,  generation_config=generation_config, pad_token_id=tokenizer.eos_token_id)
            
        time_end = timeit.default_timer()

        inference_time = time_end - time_start

        return generate_ids, inference_time, template_prompt

    def takedown_r_cad(self, prompt, gt, tokenizer, model):
        if self.config.model.setting == 'ft':
            context = ''

        null_prompt = prompt + ' ' + gt        

        template_prompt = apply_prompt_template(prompt_template_style='code', dataset=[prompt], context=context, model='code')[0]

        inputs = tokenizer(template_prompt, return_tensors="pt").to(model.device)
        null_inputs = tokenizer(null_prompt, return_tensors="pt").to(model.device)
        generation_config = GenerationConfig(do_sample=False, num_return_sequences=1, context_aware_decoding_alpha=self.config.method.context_aware_decoding_alpha, max_new_tokens=self.config.model.completion_len)

        time_start = timeit.default_timer()
        with torch.no_grad():
            generate_ids = model.generate(tokenizer, inputs=inputs.input_ids, null_inputs=null_inputs.input_ids, attention_mask=inputs.attention_mask, generation_config=generation_config, pad_token_id=tokenizer.eos_token_id)
            
        time_end = timeit.default_timer()

        inference_time = time_end - time_start

        return generate_ids, inference_time, template_prompt
    
    
    
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
        
    def takedown(self, task_id, prompt, gt, task, LLM):
        generate_ids = None
        tokenizer, model = LLM.tokenizer, LLM.model
        

        if self.config.method.intervention_name == 'r_cad':
            generate_ids, inference_time, template_prompt = self.takedown_r_cad(prompt, gt, tokenizer, model)
        elif self.config.method.intervention_name == 'FFT_r_cad':
            generate_ids, inference_time, template_prompt = self.takedown_FFT_r_cad(prompt, gt, tokenizer, model)
        elif self.config.method.intervention_name == 'speculative_r_cad':
            assistant_model = LLM.assistant_model
            generate_ids, inference_time, template_prompt = self.takedown_speculative_r_cad(prompt, gt, tokenizer, model, assistant_model)
        

        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        final_outputs, full_generated_outputs = self.postprocess(prompt, template_prompt, task, outputs)
        
        self.save_solution(task_id, full_generated_outputs, gt, task)

        return final_outputs, full_generated_outputs, inference_time
    
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
        