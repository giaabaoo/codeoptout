import os
import json
import pdb
from bigcode_eval import tasks


class CodeDataset:
    '''
    Code Dataset class for loading data from BigCode Eval
    '''
    def __init__(self, config):
        self.config = config
        self.task = tasks.get_task(self.config.dataset.name)
        self.dataset_class = self.config.dataset.dataset_class

        # Debug
        # self.items = self.task.get_dataset().select(range(117, 164))

        self.items = self.task.get_dataset()

        if self.config.dataset.num_samples:
            self.items = self.items.select(range(self.config.dataset.num_samples))

        n_tasks = len(self.items)
        self.references = [self.task.get_reference(self.items[i]) for i in range(0, n_tasks)]

        self.items = self.items.map(self.format_prompt)
        self.items = self.rename_columns()

    def rename_columns(self):
        if self.config.dataset.name == 'humaneval':
            self.items = self.items.rename_column('canonical_solution', 'gt')
        elif 'mbpp' in self.config.dataset.forget_data:
            self.items = self.items.rename_column('text', 'prompt')
            self.items = self.items.rename_column('code', 'gt')
            self.items = self.items.rename_column('test_list', 'test')


    def format_prompt(self, sample):
        if self.config.dataset.name == 'humaneval':
            sample['prompt'] = self.task.get_prompt(sample)
        elif 'mbpp' in self.config.dataset.forget_data:
            sample['text'] = self.task.get_prompt(sample)

        return sample



        