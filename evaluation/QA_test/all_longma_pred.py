import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from models.pear_longma import LongmaForCausalLM
import torch.nn as nn


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k-wo", "llama2-7b-chat-4k", "llama2-test",  "longma2-test",
    "longma2-5", "longma2-10", "longma2-15", "longma2-20", "longma2-25", "longma2-30", "longma2-40",
    "longma2", "filter_llama2_longma2_checkpoint-2250", "filter_llama2_longma2_checkpoint-250", "filter_longma2_checkpoint-250", "filter_longma2_checkpoint-270", 
    "filter_longma2_checkpoint-350", "filter_longma2_checkpoint-410", "filter_longma2_checkpoint-480", "longma2_checkpoint-100", "longma2_checkpoint-230", 
    "llama2_longma2_checkpoint-500", "longma2_checkpoint-1570"])
    parser.add_argument('--e', action='store_true', help="Evaluate")
    return parser.parse_args(args)


def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        output = model.generate(
            **input,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def set_alphas(path, model, neg_layer_filter, neg_head_filter):
    alphas = torch.load(path)
    for layeridx in neg_layer_filter:
        for idx in neg_head_filter[layeridx]:
            alpha = alphas[f'model.layers.{layeridx}.self_attn.alpha']
            model.model.layers[layeridx].self_attn.alpha = nn.Parameter(alpha)
            print(f'({layeridx},{idx}):{model.model.layers[layeridx].self_attn.alpha[idx]}')

def load_model_and_tokenizer(path, model_name, device):
    if "longma2" in model_name:
        if "5" in model_name:
            neg_layer_filter = [26, 11, 14, 30, 18]
            neg_head_filter = {26: [28], 11: [6], 14: [15], 30: [9], 18: [9]}
            alpha_path = './models/ablation_5_negfilter_50000_train_epoch1/checkpoint-60/model.bin'
        elif "10" in model_name:
            neg_layer_filter = [26, 11, 14, 30, 18, 15, 13, 12, 10]
            neg_head_filter = {26: [28], 11: [6], 14: [15], 30: [9], 18: [9], 15: [10, 14], 13: [9], 12: [10], 10: [18]}
            alpha_path = './models/ablation_10_negfilter_50000_train_epoch1/checkpoint-60/model.bin'
        elif "15" in model_name:
            neg_layer_filter = [26, 11, 14, 30, 18, 15, 13, 12, 10, 19, 29]
            neg_head_filter = {26: [28], 11: [6], 14: [0, 15], 30: [9], 18: [9], 15: [10, 14, 25], 13: [9], 12: [10], 10: [2, 18], 19: [25], 29: [15]}
            alpha_path = './models/ablation_15_negfilter_50000_train_epoch1/checkpoint-60/model.bin'
        elif "20" in model_name:
            neg_layer_filter = [26, 11, 14, 30, 18, 15, 13, 12, 10, 19, 29, 31, 8, 17, 20, 9]
            neg_head_filter = {26: [28], 11: [6], 14: [0, 15], 30: [9], 18: [9], 15: [10, 14, 25], 13: [9], 
                            12: [10], 10: [2, 18], 19: [25], 29: [15], 31: [17], 8: [22], 17: [0], 20: [26], 9: [13]}
            alpha_path = './models/ablation_20_negfilter_50000_train_epoch1/checkpoint-60/model.bin'
        elif "25" in model_name:
            neg_layer_filter = [26, 11, 14, 30, 18, 15, 13, 12, 10, 19, 29, 31, 8, 17, 20, 9, 7]
            neg_head_filter = {26: [28], 11: [6, 9], 14: [0, 15], 30: [9], 18: [9], 15: [10, 12, 14, 25], 13: [9, 14], 
                            12: [10], 10: [1, 2, 18], 19: [25], 29: [15], 31: [17], 8: [22], 17: [0], 20: [26], 9: [13], 7: [9]}
            alpha_path = './models/ablation_25_negfilter_50000_train_epoch1/checkpoint-60/model.bin'
        elif "30" in model_name:
            neg_layer_filter = [26, 11, 14, 30, 18, 15, 13, 12, 10, 19, 29, 31, 8, 17, 20, 9, 7, 28]
            neg_head_filter = {26: [9, 28], 11: [6, 9], 14: [0, 15], 30: [9], 18: [9], 15: [2, 7, 10, 12, 14, 25], 13: [9, 14], 12: [10], 10: [1, 2, 18], 
                            19: [25], 29: [15], 31: [17], 8: [22], 17: [0], 20: [26], 9: [13, 16], 7: [9], 28: [22]}
            alpha_path = './models/ablation_30_negfilter_50000_train_epoch1/checkpoint-60/model.bin'
        elif "40" in model_name:
            neg_layer_filter = [26, 11, 14, 30, 18, 15, 13, 12, 10, 19, 29, 31, 8, 17, 20, 9, 7, 28, 0, 16]
            neg_head_filter = {26: [9, 28, 19], 11: [6, 9], 14: [0, 15], 30: [9], 18: [9, 2], 15: [2, 7, 10, 11, 12, 14, 25], 13: [9, 14], 12: [10, 13, 19], 
            10: [1, 2, 3, 18], 19: [25, 27], 29: [15], 31: [17], 8: [22], 17: [0], 20: [26], 9: [13, 16, 27], 7: [9], 28: [22], 0: [25], 16: [14]}
            alpha_path = './models/ablation_40_negfilter_50000_train_epoch1/checkpoint-60/model.bin'
        
        alphas = torch.load(alpha_path, map_location=device)
        tokenizer = LlamaTokenizer.from_pretrained(path)
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        config.topk = 30
        config._attn_implementation = 'eager'
        # model = LongmaForCausalLM.from_pretrained(path, _attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        model = LongmaForCausalLM.from_pretrained(path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        '''
        In reference, we re-scale the output W_O matrix of the RAG suppression head, which is equal to re-weighting head output.
        '''
        for layer_idx in neg_layer_filter:
            head_mask = torch.zeros(32, dtype=torch.bfloat16, device=device)
            for item in neg_head_filter[layer_idx]:
                head_mask[item] = 1.0

            scales = torch.ones(32, dtype=torch.bfloat16, device=device)
            scales = torch.where(head_mask == 1.0, alphas[f'model.layers.{layer_idx}.self_attn.alpha'], scales)
            
            w = model.model.layers[layer_idx].self_attn.o_proj.weight.data.view(4096, 32, 128)
            scales = scales.view(1, 32, 1)
            w = w * scales
            model.model.layers[layer_idx].self_attn.o_proj.weight.data.copy_(w.view(4096, 4096))

    elif "llama2" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    max_length = model2maxlen[model_name]
    datasets = ["2wikimqa", "musique", "qasper"]
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    if not os.path.exists("pred"):
        os.makedirs("pred")
    for dataset in datasets:
        data = load_dataset('json', data_files=f"data/{dataset}.jsonl", split='train')
        if not os.path.exists(f"pred/{model_name}"):
            os.makedirs(f"pred/{model_name}")
        out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        get_pred(rank, world_size, data_subsets[rank], max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path)
