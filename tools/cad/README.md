#### Install the cutomized huggingface library with the change of generation scripts
```
cd transformers_cad
pip install -e .
```


#### Add context-aware decoding
replace `transformers/src/transformers/generation/utils.py` with `generation/utils.py`
