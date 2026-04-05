import os
import json
import pdb
import evaluate
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import string
from collections import defaultdict
from pathlib import Path

class CodeEvaluator:
    '''
    Base code evaluator class 
    '''
    def __init__(self, config):
        self.config = config

    def eval_infringement(self, prompt_list, gt_list, output_list, inference_time_list, save_path):  
        calculator = Calculator()
        
        # rouge = evaluate.load('rouge') HF 
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum']) # BAO        
        rouge_1, rouge_l, prompts = [], [], []

        # eval semantic similarity
        semantic_sim = []
        model = SentenceTransformer("all-MiniLM-L6-v2")

        best_rouge1_comps, best_rougeL_comps, best_verbatim_matching_comps, gts, matching_sequences = [], [], [], [], []
        best_rouge1_ids, best_rougeL_ids, best_verbatim_matching_ids = [], [], []
        best_verbatim_matching_ids, max_lengths, total_lengths = [], [], [] 
        # begin compute time
        # time_start = timeit.default_timer()
        
        for prompt, gt, outputs in zip(prompt_list, gt_list, output_list):
            outputs = [outputs] # BAO: need to be in this format!
            best_verbatim_matching_id, matching_sequence, max_length, total_length = calculator.find_common_sequences(outputs, gt)
            
            # results = rouge.compute(predictions=outputs, references=[gt]*len(outputs), use_aggregator=False) HF
            scores = scorer.score(gt, outputs[0]) # BAO
            results = {key: [value.fmeasure] for key, value in scores.items()} # BAO


            # semantic simlarity
            ref_embeddings = model.encode([gt])
            pred_embeddings = model.encode(outputs)
            cos_sim = util.cos_sim(pred_embeddings, ref_embeddings).cpu().numpy().squeeze().tolist()
            if isinstance(cos_sim, float):
                cos_sim = [cos_sim]
            max_cos_sim = max(cos_sim)
            semantic_sim.append(max_cos_sim)
            # bp()

            max_rougeL = max(results['rougeL'])
            max_rouge1 = max(results['rouge1'])
            best_rougeL = outputs[results['rougeL'].index(max_rougeL)]
            best_rouge1 = outputs[results['rouge1'].index(max_rouge1)]
            best_verbatim_matching = outputs[best_verbatim_matching_id]
            
            
            prompts.append(prompt)
            rouge_1.append(max_rouge1)
            rouge_l.append(max_rougeL)
            best_rouge1_comps.append(best_rouge1)
            best_rougeL_comps.append(best_rougeL)
            best_verbatim_matching_comps.append(best_verbatim_matching)
            best_rouge1_ids.append(results['rouge1'].index(max_rouge1))
            best_rougeL_ids.append(results['rougeL'].index(max_rougeL))
            best_verbatim_matching_ids.append(best_verbatim_matching_id)
            max_lengths.append(max_length)
            total_lengths.append(total_length)
            gts.append(gt)
            matching_sequences.append(matching_sequence)
        
        df = pd.DataFrame({'prompt': prompts, 'gt': gts, 'rouge1': rouge_1, 'rougeL': rouge_l, 'semantic_sim': semantic_sim,
                        'best_rouge1': best_rouge1_comps, 'best_rougeL': best_rougeL_comps, 'best_verbatim_matching': best_verbatim_matching_comps,
                        'matching_sequence': matching_sequences,
                        'max_length': max_lengths, 'total_length': total_lengths,
                        'best_rouge1_ids': best_rouge1_ids, 'best_rougeL_ids': best_rougeL_ids, "best_verbatim_matching_ids": best_verbatim_matching_ids, "inference_time": inference_time_list}) 
        

        try:
            df.to_excel(save_path, index=False)
        except:
            df['best_rouge1'] = df['best_rouge1'].apply(self.sanitize_string)
            df['best_rougeL'] = df['best_rougeL'].apply(self.sanitize_string)
            df['best_verbatim_matching'] = df['best_verbatim_matching'].apply(self.sanitize_string)
            df['matching_sequence'] = df['matching_sequence'].apply(self.sanitize_string)

            df.to_excel(save_path, index=False)

        print(f"Finish evaluating infringement! Results saved at: {save_path}")
    def sanitize_string(self, s):
        return ''.join(c for c in s if c.isprintable())

    def eval_code_functionality(self, task, results_dict, references, save_path):
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        
        p1_score, testing_results = task.process_results(results_dict['generation_list'], references)
        
        with open(save_path, "w") as f:
            f.write(f"{p1_score['pass@1']:.4f} \n\n {self.config}")
        

        # Convert defaultdict to a regular dict
        testing_results_dict = {key: value for key, value in testing_results.items()}

        # Separate passed and failed cases
        passed_cases = defaultdict(list)
        failed_cases = defaultdict(list)

        for key, tasks in testing_results_dict.items():
            for idx, task_info in tasks:
                if task_info['passed']:
                    passed_cases[results_dict['task_id'][key]].append(task_info) 
                else:
                    failed_cases[results_dict['task_id'][key]].append(task_info)
        

        # Define the path for passed and failed cases
        set_name = save_path.split("/")[-1].split("_")[0]
        passed_results_path = "/".join(save_path.split("/")[:-1]) + f"/{set_name}_passed_testing_results.json"
        failed_results_path = "/".join(save_path.split("/")[:-1]) + f"/{set_name}_failed_testing_results.json"

        # Save passed cases to JSON
        with open(passed_results_path, "w") as f:
            passed_cases = {key: passed_cases[key] for key in sorted(passed_cases)}
            json.dump(passed_cases, f, indent=4)

        # Save failed cases to JSON
        with open(failed_results_path, "w") as f:
            failed_cases = {key: failed_cases[key] for key in sorted(failed_cases)}
            json.dump(failed_cases, f, indent=4)
       

        print(f"Finish evaluating code functionality! Results saved at: {save_path}")

    def eval_math_functionality(self, task, generation_list, references, save_path):   
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        generation_list = [[o] for o in generation_list]

        score = task.process_results(generation_list, references)
        
        with open(save_path, "w") as f:
            f.write(f"{score['accuracy']:.4f} \n\n {self.config}")

        print(f"Finish evaluating math functionality! Results saved at: {save_path}")
    
    def evaluate(self, task, results_dict):        
        data_name = results_dict['data_name']
        set_name = results_dict['set_name']

        if data_name in  ['humaneval', 'mbpp', 'mbpp_filtered_deepseek', 'mbpp_filtered_gwen', 'mbpp_filtered_gwen_base', 'mbpp_filtered_yi']:
            if set_name == "forget":
                infringement_save_path = f'{self.config.output_dir}/{set_name}_infringement_result.xlsx'
                self.eval_infringement(results_dict['prompt_list'], results_dict['gt_list'], results_dict['output_list'], results_dict['inference_time_list'], infringement_save_path)

            functionality_save_path = f'{self.config.output_dir}/{set_name}_functionality_result.txt'
            self.eval_code_functionality(task, results_dict, results_dict['references'], functionality_save_path)

        elif data_name in ['pal-gsm8k-greedy']:
            functionality_save_path = f'{self.config.output_dir}/{set_name}_functionality_result.txt'

            self.eval_math_functionality(task, results_dict['output_list'], results_dict['references'], functionality_save_path)

       
class Calculator:
    def process_sentence(self, sentence):
        # Convert to lower case
        lower_case_sentence = sentence.lower()
        # Split the sentence into words and concatenate them
        concatenated_sentence = ''.join(lower_case_sentence.split())
        # Remove all punctuations
        final_sentence = ''.join(char for char in concatenated_sentence if char not in string.punctuation)
        return final_sentence


    def count_elements(self, lst):
        """
        count the number of the elements for multi-level list
        """
        count = 0
        for element in lst:
            if isinstance(element, list):
                count += self.count_elements(element)
            else:
                count += 1
        return count


    def QUIP(self, num_matchings, input_sequence, n=50):
        """
        n: int, n-gram size
        """
        n_grams = []
        for i in range(len(input_sequence)):
            n_grams.append(len(input_sequence[i]) - n + 1)
        result = [num / ngram for num, ngram in zip(num_matchings, n_grams)]
        return result
    
    def find_common_sequences(self, sentences1, sentence2, min_tokens=1):
        # Split the sentences into tokens
        max_lengths = []
        total_lengths = []
        common_sequences_all = []
        normalized_sentence2 = self.process_sentence(sentence2) 
        for sentence1 in sentences1:
            normalized_sentence1 = self.process_sentence(sentence1)
            # tokens1 = sentence1.split()
            # tokens2 = sentence2.split()
            tokens1 = list(normalized_sentence1)
            tokens2 = list(normalized_sentence2)

            # Create a matrix to store the lengths of common subsequences
            dp = [[0] * (len(tokens2) + 1) for _ in range(len(tokens1) + 1)]

            # Fill the matrix
            for i in range(1, len(tokens1) + 1):
                for j in range(1, len(tokens2) + 1):
                    if tokens1[i - 1] == tokens2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = 0  # Change to 0 to reset the count for non-continuous matches

            # Find all common sequences with their start and end indices
            sequences_with_indices = []
            for i in range(1, len(tokens1) + 1):
                for j in range(1, len(tokens2) + 1):
                    if dp[i][j] >= min_tokens:
                        sequence = ' '.join(tokens1[i - dp[i][j]:i])
                        sequences_with_indices.append(((i - dp[i][j], i - 1), sequence))

            # Filter out overlapping sequences to keep only the longest ones
            longest_sequences = []
            sequences_with_indices.sort(key=lambda x: (x[0][0], -x[0][1]))  # Sort by start index and then by reverse end index
            last_end_index = -1
            longest_sequence_length = 0
            total_length = 0
            for indices, sequence in sequences_with_indices:
                if indices[0] > last_end_index:
                    longest_sequences.append(sequence)
                    last_end_index = indices[1]
                    sequence_length = len(sequence.split())
                    longest_sequence_length = max(longest_sequence_length, sequence_length)
                    total_length += sequence_length

            max_lengths.append(longest_sequence_length)
            total_lengths.append(total_length)
            common_sequences_all.append(longest_sequences)
        
        top_overlapping_id = total_lengths.index(max(total_lengths))

        return top_overlapping_id, common_sequences_all[top_overlapping_id], max_lengths[top_overlapping_id], total_lengths[top_overlapping_id]




        
