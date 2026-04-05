import os
import json
import pdb
import timeit
import torch
from pathlib import Path
from .tool.prompt_utils import apply_prompt_template
import re

class KETakedownMethod:
    '''
    Base class for taking down copyrights based on training-based strategy on a single sample (knowledge editing)
    '''
    def __init__(self, config):
        self.config = config        

    # takedown without returning, change LLM: used for training time takedown
    def takedown_only(self, prompt, gt, LLM):
        self.locate_then_edit(prompt, gt, LLM)

    # takedown with returning, no change LLM
    def takedown(self, task_id, prompt, gt, task, LLM):
        self.locate_then_edit(prompt, gt, LLM)
        
        final_outputs, full_generated_outputs, inference_time = self.inference(task_id, prompt, gt, task, LLM)

        return final_outputs, full_generated_outputs, inference_time
    
    # def inference_only(self, task_id, prompt, gt, task, tokenizer, model):
    #     final_outputs, full_generated_outputs, inference_time = self.inference(task_id, prompt, gt, task, tokenizer, model)

    #     return final_outputs, full_generated_outputs, inference_time
    
    
    def locate_then_edit(self, prompt, gt, LLM):
        highly_activated_layers = self.find_highly_activated_layers(prompt, gt, LLM)
        self.edit(prompt, gt, LLM, highly_activated_layers)
        

    def find_highly_activated_layers(self, prompt, gt, LLM):
        # Method 1: Compare prompt embeddings and gt embeddings!
        prediction = prompt

        # Method 2:  Compare answer embeddings and gt embeddings!
        # inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        # # Generate code using the model
        # with torch.no_grad():
        #     generated_ids = model.generate(inputs['input_ids'], max_new_tokens=self.config.model.completion_len, do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=inputs.attention_mask) # None but greedy decoding

        # prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # prediction = prediction.replace(prompt, "")

        ####### LOCATING ######
        # Tokenize the prompt and ground truth
        inputs = LLM.tokenizer([prediction, gt], return_tensors="pt", padding=True).to(LLM.model.device)

        with torch.no_grad():
            outputs = LLM.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states

        
        distances = []
            

        for layer_index in range(1, len(hidden_states)):
            layer_hidden_states_prompt_output = hidden_states[layer_index][0]
            layer_hidden_states_gt = hidden_states[layer_index][1]

            # Calculate the Euclidean distance between prompt and ground truth hidden states
            euclidean_distance = torch.dist(layer_hidden_states_prompt_output, layer_hidden_states_gt, p=2)
            distances.append((layer_index - 1, euclidean_distance.item()))

        distances.sort(key=lambda x: x[1])
        top_k = min(self.config.method.top_k, len(distances))

        return [layer for layer, _ in distances[:top_k]]

    def edit(self, prompt, gt, LLM, highly_activated_layers): 
        # Prepare the data
        input_ids, targets = self.format_data(prompt, gt, LLM) 

        # Set up the optimizer to only update parameters of specific layers
        optimizer_params = []

        # Define the layer index to be fine-tuned

        for name, param in LLM.model.named_parameters():
            parts = name.split('.')
            if len(parts) > 2 and parts[2].isdigit():
                layer_index = int(parts[2])
                if layer_index in highly_activated_layers:
                    layer_name = parts[3]
                    if self.config.method.layer_type in layer_name:
                        param.requires_grad = True
                        optimizer_params.append(param)
                    elif self.config.method.layer_type == "all":
                        param.requires_grad = True
                        optimizer_params.append(param)
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

        
        # # DEBUG
        # print("Layer Name | requires_grad")
        # print("-" * 30)
        # for name, param in model.named_parameters():
        #     print(f"{name} | {param.requires_grad}")


        optimizer = torch.optim.Adam(optimizer_params, lr=float(self.config.method.learning_rate))

        if self.config.method.edit_method == "GA":
            self.GA(input_ids, targets, LLM, optimizer)

    
    def format_data(self, prompt, targets, LLM):
        # Combine prompt and targets
        full_text = prompt + targets

        # Tokenize the combined text
        inputs = LLM.tokenizer(full_text, add_special_tokens=False)
        input_ids = inputs['input_ids'] + [LLM.tokenizer.eos_token_id] # 25-10-2024: remove [LLM.tokenizer.bos_token_id] +
        attention_masks = [1] + inputs['attention_mask'] + [1]

        num_prompt_tokens = len(LLM.tokenizer(prompt, add_special_tokens=True)['input_ids']) 

        label_ids = input_ids.copy()
        # Change label to -100 for question tokens
        for i in range(num_prompt_tokens):
            label_ids[i] = -100
        
        # Shift +1 for the label since we are doing next word prediction
        label_ids = label_ids[1:] + [-100]


        input_ids = {'input_ids': torch.tensor(input_ids).to(LLM.model.device).unsqueeze(0), 
                     'attention_mask': torch.tensor(attention_masks).to(LLM.model.device).unsqueeze(0)}
        label_ids = torch.tensor(label_ids).to(LLM.model.device)

        return input_ids, label_ids

    def get_loss(self, output, labels):
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_function(output.view(-1, output.size(-1)), labels.view(-1))

        return loss
    
    def GA(self, input_ids, targets, LLM, optimizer):
        # Training loop
        LLM.model.train()

        for epoch in range(self.config.method.num_epochs): # default: unlearning 1 epoch
            optimizer.zero_grad()
            outputs = LLM.model(**input_ids)
            logits = outputs.logits
    
            forget_loss = self.get_loss(logits, targets)
            loss = forget_loss * -1

            loss.backward()
            optimizer.step()
                
    def inference(self, task_id, prompt, gt, task, LLM):
        template_prompt = apply_prompt_template(prompt_template_style='code', dataset=[prompt], context="", model='code')[0]
        inputs = LLM.tokenizer(template_prompt, return_tensors="pt").to(LLM.model.device)
        time_start = timeit.default_timer()
        generate_ids = LLM.model.generate(inputs.input_ids, max_new_tokens=self.config.model.completion_len, do_sample=False, num_return_sequences=1, pad_token_id=LLM.tokenizer.eos_token_id, attention_mask=inputs.attention_mask) # None but greedy decoding
        time_end = timeit.default_timer()

        inference_time = time_end - time_start
        outputs = LLM.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)  
        
        
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



    # BAO: deprecated
    def batch_padding(self, prompt, targets, tokenizer, max_length, model):
        # Combine prompt and targets
        full_text = prompt + targets

        # Tokenize the combined text
        encoded = tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True
        )

        num_prompt_tokens = len(tokenizer.tokenize(prompt, add_special_tokens=False))
        pad_length = max_length - len(encoded.input_ids)

        # Padding the input ids and attention mask
        pad_input_ids = encoded['input_ids'] + [tokenizer.pad_token_id] * pad_length
        pad_attention_mask = encoded['attention_mask'] + [0] * pad_length

        # Create labels with -100 for padding tokens
        if len(encoded.input_ids) == max_length:
            label = encoded.input_ids
        else:
            label = encoded['input_ids'] + [tokenizer.pad_token_id] + [-100] * (pad_length - 1)

        # Change label to -100 for question tokens
        for i in range(num_prompt_tokens):
            label[i] = -100


        input_ids = {'input_ids': torch.tensor(pad_input_ids).to(model.device).unsqueeze(0), 
                     'attention_mask': torch.tensor(pad_attention_mask).to(model.device).unsqueeze(0)}
        label = torch.tensor(label).to(model.device)

        return input_ids, label


