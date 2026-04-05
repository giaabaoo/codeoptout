from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from argparse import ArgumentParser
import pdb
from transformers import AutoTokenizer, AutoModelForCausalLM


from bigcode_eval import tasks
import pandas as pd
import datasets

from huggingface_hub import login
login(token="YOUR_HF_TOKEN")

DATA_COL = {
    "inst": "instruction",
    "output": "output"
}

EOS_TOKEN = None

def formatting_prompts_func(examples):
    prompt =  "{}\n{}"

    instructions = examples['instruction']
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # text = prompt.format(instruction, output)
        text = prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }



def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--data_split", type=str, default="full")
    args = parser.parse_args()


    max_seq_length = 500 # Supports RoPE Scaling interally, so choose any!

    # Get dataset
    url = args.data_path

    if 'humaneval' in url:
        DATA_COL["inst"] = 'prompt'
        DATA_COL["output"] = 'canonical_solution'
    elif 'mbpp' in url:
        DATA_COL["inst"] = 'text'
        DATA_COL["output"] = 'code'

    task = tasks.get_task(args.data_path)
    dataset = task.get_dataset()
    n_tasks = len(dataset)
    prompts = [task.get_prompt(example) for example in dataset]
    samples = [{"instruction": q, "output": a} for q, a in zip(prompts, dataset[DATA_COL["output"]])]
    if args.data_split == 'retain':
        percent = int(args.data_split.replace("retain", ""))
        n_forget = (n_tasks * percent) // 100
        samples = samples[n_forget:n_tasks]

    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=samples))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = max_seq_length,
        dtype = torch.bfloat16,
        load_in_4bit = True,
    )
    global EOS_TOKEN
    EOS_TOKEN = tokenizer.eos_token
    
    dataset = dataset.map(formatting_prompts_func, batched = True)

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        max_seq_length = max_seq_length,
    )

    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        tokenizer = tokenizer,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 120,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            seed = 3407,
            learning_rate = 2e-4,
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            output_dir = args.output_dir,
        ),
    )
    trainer.train()

    # Go to https://github.com/unslothai/unsloth/wiki for advanced tips like
    # (1) Saving to GGUF / merging to 16bit for vLLM
    # (2) Continued training from a saved LoRA adapter
    # (3) Adding an evaluation loop / OOMs
    # (4) Customized chat templates

    model.save_pretrained_merged(args.output_dir, tokenizer, 'merged_16bit')
    # if args.push_model:
    #     model.push_to_hub_merged(args.save_path, tokenizer, args.hub_token)


if __name__ == '__main__':
    main()