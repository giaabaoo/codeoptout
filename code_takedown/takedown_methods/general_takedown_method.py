import os
import json
import pdb
from .tool.prompt_utils import apply_prompt_template
from .tool.decoding_intervention import TopKPerturbationLogitsProcessor
import timeit
from pathlib import Path

class GeneralTakedownMethod:
    '''
    Base method class for taking down copyrights based on general strategy
    '''
    def __init__(self, config):
        self.config = config

    def takedown_top_k(self, prompt, gt, tokenizer, model):
        if self.config.model.setting == 'ft':
            context = ''
        elif self.config.model.setting == 'rag':
            context = f"Copyrighted code: {gt}\n" + 'Please complete the following code: \n' + prompt
       
        prompt = apply_prompt_template(prompt_template_style='code', dataset=[prompt], context=context, model='code')[0]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        time_start = timeit.default_timer()
        generate_ids = model.generate(tokenizer, inputs.input_ids, max_new_tokens=self.config.model.completion_len, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask)
        time_end = timeit.default_timer()

        inference_time = time_end - time_start

        return generate_ids, inference_time, prompt

    def takedown_sys_prompt(self, prompt, gt, tokenizer, model):
        if self.config.model.setting == 'rag':
            context = f"Copyrighted code: {gt}\n" + 'Please complete the following code: \n' + prompt
        else:
            context = ''

        system_prompt_choice = self.config.method.intervention_name.split('-')[-1]
        template_prompt = apply_prompt_template(prompt_template_style=system_prompt_choice, dataset=[prompt], context=context, model='code')[0]

        inputs = tokenizer(template_prompt, return_tensors="pt").to(model.device)
        time_start = timeit.default_timer()
        generate_ids = model.generate(tokenizer, inputs.input_ids, max_new_tokens=self.config.model.completion_len, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask) # The only difference is do_sample
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

        if self.config.method.intervention_name == 'top_k':
            generate_ids, inference_time, template_prompt = self.takedown_top_k(prompt, gt, tokenizer, model)
        elif 'sys_prompt' in self.config.method.intervention_name:
            generate_ids, inference_time, template_prompt = self.takedown_sys_prompt(prompt, gt, tokenizer, model)
        
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        final_outputs, full_generated_outputs = self.postprocess(prompt, template_prompt, task, outputs)
        # pdb.set_trace()
        
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
        