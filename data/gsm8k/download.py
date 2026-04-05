import datasets

ds = datasets.load_dataset("gsm8k", 'main') 
ds.save_to_disk(".")