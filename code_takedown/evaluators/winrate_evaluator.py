import os
import pdb
import evaluate
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import string
from pathlib import Path
from datasketch import MinHash
import re
import numpy as np
import Levenshtein
from tqdm import tqdm
from codebleu import calc_codebleu

import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

from .code_style_sim import *
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

class WinrateEvaluator:
    '''
    Base winrate evaluator class 
    '''
    def __init__(self, config):
        self.config = config

        # Load the CodeSage model and tokenizer for code emb similarity
        self.checkpoint = "codesage/codesage-small"  # or 'codesage/codesage-small', etc.
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, add_eos_token=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.checkpoint, trust_remote_code=True).to("cuda")
        PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser(PY_LANGUAGE)   

    def process_infringement_results(self):
        in_file_paths_list = self.config.evaluator.file_list       
        out_file_paths_list = []
        for file in tqdm(in_file_paths_list):
            # in_file_path = os.path.join("./", self.config.output_dir.split("/")[1], file, "infringement_result.xlsx")
            # out_file_path = os.path.join(self.config.output_dir.split("/")[1], file, "processed_infringement_result.xlsx")
            in_file_path = "./results/" + file
            out_file_path = "./results/" + "/".join(s for s in file.split("/")[:-1]) + "/" + "processed_infringement_result.xlsx" 

            if os.path.exists(out_file_path):
                out_file_paths_list.append(out_file_path)
                continue
      
            df = pd.read_excel(in_file_path)
            processed_df = self.add_metrics(df)
            processed_df.to_excel(out_file_path, index = False)
            out_file_paths_list.append(out_file_path)

        return out_file_paths_list

    def evaluate(self):
        file_list = self.process_infringement_results()
        
        if self.config.model.setting == "rag":
            df = self.win_rate_rag(file_list)
            save_path = f"{self.config.output_dir}/win_rate_rag_results.xlsx"
            df.to_excel(save_path, index=False)
        elif self.config.model.setting == "ft":
            df = self.win_rate_ft(file_list)
            save_path = f"{self.config.output_dir}/win_rate_ft_results.xlsx"
            df.to_excel(save_path, index=False)

    def win_rate_ft(self, file_list):
        method_data_frames = []
        methods = {} # Dictionary containing takedown method name as key and value being the win rate count
        metrics = self.config.evaluator.metrics
        for file in file_list:
            df = pd.read_excel(file)
            method_data_frames.append(df)
            method_name = self.extract_intervention(file)
            methods[method_name] = {metric: 0 for metric in metrics}

        try:
            num_samples = len(method_data_frames[0])
        except:
            print("The numbers of samples in these files are mis-matched")

        # DEBUG code
        # count_rouge_1_draw = 0
        # count_rouge_1_vanilla_win = 0
        # count_rouge_1_unlearn_win = 0
        # r1_draw_set = set()

        for i in range(num_samples):
            concatenated_df = pd.DataFrame([df.iloc[i] for df in method_data_frames])
            concatenated_df['method'] = list(methods.keys())

            win_rate_df = pd.DataFrame()
            win_rate_df['method'] = concatenated_df ['method']


            for metric in metrics:
                win_rate_df[metric + '_win_rate'] = self.compute_win_rate(concatenated_df, metric)
            
            
            # Get 8 columns (methods) names
            win_rate_count_rows_to_select = [metric + '_win_rate' for metric in metrics]
            win_rate_df = win_rate_df.reset_index(drop=True)

            for method in methods.keys():
                # win_rate_count_method = a list containing win rate count for each method [0,0, etc.]
                win_rate_count_method = win_rate_df.loc[win_rate_df['method'] == method, win_rate_count_rows_to_select].values.tolist()[0]
                
                for metric, y in zip(methods[method], win_rate_count_method):
                    methods[method][metric] = methods[method][metric] + y
                    # add the number of times a method win in a metric 

                    # DEBUG code
                    # if y > 0 and method == "unlearn_GA" and metric == "rouge1":
                    #     count_rouge_1_unlearn_win += 1
                    # elif y > 0 and method == "vanilla" and metric == "rouge1":
                    #     count_rouge_1_vanilla_win +=1
                    # --> this elif count the number of draws!
            
            # if win_rate_df["rouge1_win_rate"].iloc[0] == win_rate_df["rouge1_win_rate"].iloc[1]:
            #     count_rouge_1_draw += 1 
            #     r1_draw_set.add(i)
            
            

        total_numbers = len(metrics) * num_samples
        
        data_columns = metrics.copy()
        data_columns.insert(0, 'methods')
        data_columns.append('average')
        data = pd.DataFrame(columns=data_columns)

        data['methods'] = list(methods.keys())
        
        for metric in metrics:
            data[metric] = [methods[method][metric]/num_samples for method in methods.keys()]

        # win_count_method_1 / num_samples + ... + win_count_method_n / num_samples --> we take the sum then divide by total_numbers = metrics * num_samples!
        data['average'] = [sum(methods[method].values())/total_numbers for method in methods.keys()]

        return data

    def win_rate_rag(self, file_list):
        return
    
    def extract_intervention(self, text):
        # Define a regex pattern to match the 'Intervention' part
        match = re.search(r'Intervention_([\w]+)', text)
    
        # Return the captured group if a match is found
        return match.group(1) if match else None

    def remove_docstring_and_tags(self, text):
        # Remove text inside [INST] and [/INST] tags
        text = re.sub(r"\[INST\].*?\[\/INST\]", "", text, flags=re.DOTALL)

        # Remove docstring (triple quotes)
        text = re.sub(r'"""""".*?""""""', '', text, flags=re.DOTALL)
        text = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
        
        return text.strip()

    def compute_win_rate(self, df, metric):
        # win rate: how many times this method win against other methods / total number of other methods!
        values = df[metric]  # Example, metric = ROUGE1, values is a list containing rouge score for each method
        win_rates = []
        for value in values:
            if 'Levenshtein Distance' in metric:
                win_rate = sum(value > other_value for other_value in values) / (len(values) - 1)
            else:
                win_rate = sum(value < other_value for other_value in values) / (len(values) - 1)
            win_rates.append(win_rate)
        return win_rates

    def compute_min_indicator(self, df, metric):
        if 'Levenshtein Distance' in metric:
            max_value = df[metric].max()
            return [1 if value == max_value else 0 for value in df[metric]] 
        else:
            min_value = df[metric].min()
            return [1 if value == min_value else 0 for value in df[metric]] 
    
    # Function to get embeddings for a code snippet
    def get_embedding(self, code_snippet):
        inputs = self.tokenizer.encode(code_snippet, return_tensors="pt").to("cuda")
        with torch.no_grad():
            embedding = self.model(inputs)[0]
        # Average pooling the embedding along the token dimension to get a single vector
        return embedding.mean(dim=1)

    def compute_code_similarity(self, code_snippet_1, code_snippet_2):
        # Get embeddings for both code snippets
        embedding_1 = self.get_embedding(code_snippet_1)
        embedding_2 = self.get_embedding(code_snippet_2)

        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(embedding_1, embedding_2)

        return cosine_sim.item()

    def add_metrics(self, res):
        columns_to_remove = ['best_rouge1', 'best_rougeL', 'matching_sequence', 'best_rouge1_ids', 'best_rougeL_ids', 'best_verbatim_matching_ids', 'total_length', 'inference_time']
        res = res.drop(columns=columns_to_remove)
        res = res.rename(columns={'max_length': 'LCS(character)'})
        res = res.rename(columns={'best_verbatim_matching': 'output'})
        columns = list(res.columns)
        columns[3], columns[4], columns[5], columns[6] = columns[6], columns[3], columns[4], columns[5]
        res = res[columns]

        # Compute word level LCS and ACS, Levenstein distance and Minhash
        lcs_word, acs_word, levenshtein_distance, min_hash_similarity, code_emb_similarity, codebleu_scores, css_scores = [], [], [], [], [], [], []
        gts = res['gt']
        outputs = res['output']

        var_idf = cal_idf(outputs, gts, 'var', "python", self.parser)
        api_idf = cal_idf(outputs, gts, 'api', "python", self.parser)
        

        for j in range(len(gts)):
            output = str(outputs[j])
            gt = gts[j]
            prompt = self.remove_docstring_and_tags(res['prompt'][j])
            _, _, max_length, total_length = self.find_common_sequences([output], gt)
            levenshtein_dist = Levenshtein.distance(output, gt)
            min_hash_sim = self.compute_min_hash_similarity(output, gt)
            emb_sim = self.compute_code_similarity(output, gt)

            #codebleu
            # function return 'codebleu', 'ngram_match_score', 'weighted_ngram_match_score', 'syntax_match_score', 'dataflow_match_score'
            code_output = prompt + output
            code_gt = prompt + gt
            codebleu_score =  calc_codebleu([code_output], [code_gt], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)['codebleu']
            codebleu_scores.append(codebleu_score)

            # code style similarity
            css_score = get_overall_csd(code_output, code_gt, "python", var_idf, api_idf)
            css_scores.append(css_score)

            lcs_word.append(max_length)
            acs_word.append(total_length)
            levenshtein_distance.append(levenshtein_dist)
            min_hash_similarity.append(min_hash_sim)
            code_emb_similarity.append(emb_sim)

            
        res['LCS(word)'] = lcs_word
        res['ACS(word)'] = acs_word
        res['Levenshtein Distance'] = levenshtein_distance
        res['Minhash Similarity'] = min_hash_similarity
        res['Code Emb Similarity'] = code_emb_similarity
        res['Codebleu Score'] = codebleu_scores
        res['CSS Score'] = css_scores
        
        return res
    
    def process_sentence(self, sentence):
        # Convert to lower case
        sentence = sentence.lower()
        translator = str.maketrans('', '', string.punctuation)
        sentence = sentence.translate(translator)
        words = sentence.split()
        return words


    def find_common_sequences(self, sentences1, sentence2, min_tokens=3):
        # Split the sentences into tokens
        max_lengths = []
        total_lengths = []
        common_sequences_all = []
        for sentence1 in sentences1:
            tokens1 = self.process_sentence(sentence1)
            tokens2 = self.process_sentence(sentence2)

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


    def compute_min_hash_similarity(self, sentence1, sentence2):
        # Convert sentences to sets of 3-grams
        def shingles(sentence, k=3):
            return {sentence[i:i+k] for i in range(len(sentence) - k + 1)}
        
        shingles1 = shingles(sentence1)
        shingles2 = shingles(sentence2)

        # Initialize MinHash objects
        m1, m2 = MinHash(), MinHash()

        # Update MinHash objects with shingles
        for shingle in shingles1:
            m1.update(shingle.encode('utf8'))

        for shingle in shingles2:
            m2.update(shingle.encode('utf8'))

        # Compute Jaccard similarity
        similarity = m1.jaccard(m2)
        return similarity