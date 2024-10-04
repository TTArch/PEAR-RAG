import torch
import pickle
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import tqdm
from tqdm import tqdm

torch.manual_seed(42)
random.seed(42)

def gen(tokenizer, num_samples, seq_len):
    for _ in tqdm(range(num_samples), desc="Generating samples"):
        input_ids = torch.randint(2, len(tokenizer), (seq_len,)).tolist()
        repeat_part = input_ids[-seq_len:]
        input_ids.extend(repeat_part)

        # labels = [-100] * seq_len + repeat_part[1:] + [-100]
        labels = [-100] * seq_len + repeat_part
        yield {"input_ids": input_ids, "labels": labels}

n_gpus = torch.cuda.device_count()
tokenizer = AutoTokenizer.from_pretrained('/home/public/meta-llama/Llama-2-7b-chat-hf', use_fast=False, padding_side="left")

num_samples = 50000
seq_len = 50

data = list(gen(tokenizer, num_samples, seq_len))
print(data)

# with open("./randomdatasets/dataset_test.pkl", "wb") as f:
#     pickle.dump(data, f)
