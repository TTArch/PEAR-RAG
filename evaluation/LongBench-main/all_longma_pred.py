import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
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
    "llama2_longma2_checkpoint-500", "longma2_checkpoint-1570", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", 
    "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "longma2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        # if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
        #     prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
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
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "longma2" in model_name:
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
        replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
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
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        #             "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        #             "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        #             "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \]
        # datasets = ["hotpotqa", "qasper", "multifieldqa_en", "gov_report", "triviaqa"]
        datasets = ["hotpotqa", "2wikimqa", "musique"]
        datasets = ["2wikimqa", "musique", "qasper"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('json', data_files=f"data/{dataset}.jsonl", split='train')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model2path, out_path))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
