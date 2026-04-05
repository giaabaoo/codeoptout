import os
import json
import pdb
from bigcode_eval import tasks


class CodeSplitDataset:
    '''
    Code Split Dataset class for loading data from BigCode Eval
    '''
    def __init__(self, config):
        self.config = config
        self.forget_task = tasks.get_task(self.config.dataset.forget_data)
        self.data_path = self.forget_task.DATASET_LOCAL_PATH
        # print("data path: " + self.data_path)
        self.dataset_class = self.config.dataset.dataset_class

        # Debug 233 234 
        # self.items = self.forget_task.get_dataset().select(range(8, 10))
        self.items = self.forget_task.get_dataset()

        if self.config.dataset.num_samples:
            self.items = self.items.select(range(self.config.dataset.num_samples))

        n_tasks = len(self.items)
        self.references = [self.forget_task.get_reference(self.items[i]) for i in range(0, n_tasks)]

        # humaneval and mbpp have different prompt format
        # self.items = self.items.map(self.format_prompt) --> does not work
        self.items = self.items.map(lambda sample: self.format_prompt(sample))

        self.rename_columns()

        self.n_forget = int((n_tasks * self.config.dataset.percent) // 100)
        self.forget_items = self.items.select(range(0, self.n_forget))

        if not (self.config.pipeline.name == "get_forget_gt_vanilla" or 'filtered' not in self.config.dataset.forget_data):
            self.forget_items = self.forget_items.rename_columns({'gt': 'real_gt'})

            # Add a new feature 'gt' with values from vanilla forget output list
            forget_path = f'data/mbpp_filtered_{self.config.model.name}/forget_data.json'
            with open(os.path.join(os.getcwd(), forget_path), 'r') as forget_file:
                forget_data = json.load(forget_file)
            forget_gt_list = forget_data['output_list']
            # Add the new column to the dataset
            assert len(forget_gt_list) == len(self.forget_items), f"Error: The length of the list does not match the number of rows in the forget dataset\n forget_gt_list: {len(forget_gt_list)}\n self.forget_items: {len(self.forget_items)}"
            self.forget_items = self.forget_items.add_column('gt', forget_gt_list)

        self.forget_references = self.references[:self.n_forget]

        if self.config.dataset.retain_data == self.config.dataset.forget_data:
            self.retain_items = self.items.select(range(self.n_forget, len(self.items)))
            self.retain_references = self.references[self.n_forget:]
            self.retain_task = self.forget_task
        else:
            self.retain_task = tasks.get_task(self.config.dataset.retain_data)
            self.retain_items = self.retain_task.get_dataset().select(range(0, len(self.forget_items)))
            self.retain_references = [self.retain_task.get_reference(self.retain_items[i]) for i in range(0, len(self.retain_items))][:self.n_forget]
            self.retain_items = self.retain_items.rename_column('question', 'prompt')
            self.retain_items = self.retain_items.rename_column('answer', 'gt')

            self.retain_items = self.retain_items.add_column('test', ['NA']*len(self.retain_items))
            self.retain_items = self.retain_items.add_column('task_id', list(range(len(self.retain_items))))

    def rename_columns(self):
        if self.config.dataset.forget_data == 'humaneval':
            self.items = self.items.rename_column('canonical_solution', 'gt')
        elif 'mbpp' in self.config.dataset.forget_data:
            self.items = self.items.rename_column('text', 'prompt')
            self.items = self.items.rename_column('code', 'gt')
            self.items = self.items.rename_column('test_list', 'test')  
        return  
        


    def format_prompt(self, sample):
        if self.config.dataset.forget_data == 'humaneval':
            sample['prompt'] = self.forget_task.get_prompt(sample)
        elif 'mbpp' in self.config.dataset.forget_data:
            sample['text'] = self.forget_task.get_prompt(sample)

        return sample



        