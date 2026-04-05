from code_takedown.utils import setup_logger
from code_takedown.datasets import get_data_loader
from code_takedown.models import get_model
from code_takedown.evaluators import get_evaluator
from tqdm import tqdm

import os
import json
import pdb

class GetForgetGT:
    def __init__(self, config):
        self.config = config

    def run(self):
        setup_logger(self.config)

        data_loader = get_data_loader(self.config)
        LLM = get_model(self.config)
        forget_references, retain_references = data_loader.forget_references, data_loader.retain_references
        
        evaluator = get_evaluator(self.config)

        
        if self.config.dataset.eval_forget:
            ################## INFERENCE ON FORGET SET ##################
            forget_task = data_loader.forget_task

            forget_results_dict = {
                'data_name': self.config.dataset.forget_data,
                'set_name': 'forget',
                'task_id': [],
                'prompt_list': [],
                'gt_list': [],
                'references': forget_references,
                'output_list': [],
                'inference_time_list': [],
                'test_case_list': [],
                'generation_list': []
            }

            for sample in tqdm(data_loader.forget_items):
                task_id, prompt, gt, test_case = sample['task_id'], sample['prompt'], sample['gt'], sample['test']
                forget_results_dict['prompt_list'].append(prompt)
                forget_results_dict['gt_list'].append(gt)
                forget_results_dict['test_case_list'].append(test_case)
            
                outputs, full_generated_outputs, inference_time = LLM.inference(task_id, prompt, gt, forget_task)      

                forget_results_dict['output_list'].append(outputs)
                forget_results_dict['inference_time_list'].append(inference_time)
                forget_results_dict['generation_list'].append(full_generated_outputs)
                forget_results_dict['task_id'].append(task_id)

                
            ################## INFERENCE ON FORGET SET ##################

            ################## EVALUATE ON FORGET SET ##################
            evaluator.evaluate(forget_task, forget_results_dict)
            ################## EVALUATE ON FORGET SET ##################
        print("Time taken for inference on forget set: ", sum(forget_results_dict['inference_time_list']))

        
        # Save forget data to eval other methods
        saved_path = os.path.join(forget_task.DATASET_LOCAL_PATH, "forget_data.json")
        with open(saved_path, "w") as json_file:
            json.dump(forget_results_dict, json_file, indent=4)
        print("========= SAVED FORGET DATA ==========\nExit...")
        return



            


        