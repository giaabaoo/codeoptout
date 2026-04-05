import datasets
ds = datasets.load_dataset("mbpp") 
ds.save_to_disk(".")
