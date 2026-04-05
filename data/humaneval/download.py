import datasets

ds = datasets.load_dataset("openai_humaneval") 
ds.save_to_disk(".")

