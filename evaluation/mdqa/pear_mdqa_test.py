# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import torch.nn as nn

from ..models.pear_longma import LongmaForCausalLM
# from llama_2.ablation_10_LongMa import LongmaForCausalLM
# from llama_2.ablation_15_LongMa import LongmaForCausalLM
# from llama_2.ablation_20_LongMa import LongmaForCausalLM
# from llama_2.ablation_25_LongMa import LongmaForCausalLM
# from llama_2.ablation_30_LongMa import LongmaForCausalLM

# from ..model.llama_2.LongMa from LongmaForCausalLM

from tqdm import tqdm
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
torch.set_printoptions(profile="full")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    topk : int = field(default=1)
    expert_nums : int = field(default=7)
    model_name: Optional[str] = field(default="longma", metadata={"help": "Model name for specific configurations"})


qa_prompt = '''Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).

{search_results}

Question: {question}
Answer:'''

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    test_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    conv_template: str = field(
        default=None, metadata={"help": "Template used to format the training data."}
    )
    lazy_preprocess: bool = False
    start: int = -1
    end: int =-1
    correct_doc_id: int = 1


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    source_model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Original maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Expanded maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)




def get_prompt(query):
    prompt=f"<s>[INST]\n{query}\n[/INST]\n"
    return prompt

def get_prompt1(query):
    prompt=f"<s>{query}\n"
    return prompt

def get_qa_prompt_index(
    question, documents, mention_random_ordering=False, query_aware_contextualization=False, answer_idx=0
):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    if not documents:
        raise ValueError(f"Provided `documents` must be truthy, got: {documents}")

    if mention_random_ordering and query_aware_contextualization:
        raise ValueError("Mentioning random ordering cannot be currently used with query aware contextualization")

#     if mention_random_ordering:
#         prompt_filename = "qa_ordered_randomly.prompt"
#     elif query_aware_contextualization:
#         prompt_filename = "qa_with_query_aware_contextualization.prompt"
#     else:
#         prompt_filename = "qa.prompt"

#     with open(PROMPTS_ROOT / prompt_filename) as f:
#         prompt_template = f.read().rstrip("\n")
    prompt_template = qa_prompt

    # Format the documents into strings
    gold_index = 0
    for document_index, document in enumerate(documents):
        if document['isgold']:
            gold_index = document_index
            break

    formatted_documents = []
    for document_index, document in enumerate(documents):
        if document['isgold']: continue
        formatted_documents.append(f"Document [{document_index+1}](Title: {documents[document_index]['title']}) {documents[document_index]['text']}")

    corrent_documents = [f"Document [{gold_index+1}](Title: {documents[gold_index]['title']}) {documents[gold_index]['text']}"]
    output_documents = formatted_documents[:answer_idx] + corrent_documents + formatted_documents[answer_idx:]

    return prompt_template.format(question=question, search_results="\n".join(output_documents))

def preprocess(
    data,
    tokenizer: transformers.PreTrainedTokenizer,
    max_length,
    correct_doc_id,
) -> Dict:
    
    question = data["question"]
    answers = data["answers"]
    docs = data['ctxs']
    query = get_qa_prompt_index(question=question,documents=docs,answer_idx=correct_doc_id)
    query = get_prompt(query)
    query_ids = tokenizer.encode(query,
                    max_length=max_length,
                    add_special_tokens=False,
                    truncation=False,
                    padding=False)


    input_ids = torch.LongTensor([query_ids])
    
    return dict(
        input_ids=input_ids,
        answers=answers,
        question= question
    )



class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_length=8192,correct_doc_id=1):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.max_length = max_length
        self.correct_doc_id = correct_doc_id
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
    
        ret = preprocess(self.raw_data[i], self.tokenizer, self.max_length,self.correct_doc_id)
        if i == 0:
            print("data case")
            print(ret['input_ids'].tolist())

        ret = dict(
            input_ids=ret["input_ids"],
            answers=ret["answers"],
            question=ret['question']
        )
        self.cached_data_dict[i] = ret

        return ret

def check_in_answers(output,answers):
    for ans in answers:
        if ans in output:
            return True
    return False

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args,max_length,correct_doc_id
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset
    rank0_print("Loading data...")
    with open(data_args.test_data_path,'r') as t:
        test_data = t.readlines()
    test_data = [json.loads(i) for i in test_data]
    test_dataset = dataset_cls(test_data, tokenizer=tokenizer,max_length=max_length,correct_doc_id=correct_doc_id)
    return test_dataset

def train():
    global local_rank
    device = torch.device("cuda:0")
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    #make_dataset
    test_dataset=make_supervised_data_module(tokenizer=tokenizer, data_args=data_args,max_length=training_args.model_max_length,correct_doc_id=data_args.correct_doc_id)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
    model_name = model_args.model_name
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
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path,trust_remote_code=True)
    config._attn_implementation = 'eager'
    config.topk = 30,
    model = LongmaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        use_safetensors=False,
    ).to(device)
    
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
    
    model.eval()
    counts = 0
    outs= []
    lens = []
    for idx,d in tqdm(enumerate(test_dataset)):
        if idx < data_args.start or idx >=data_args.end:
            continue
        answers = d['answers']
        question = d['question']
        input_ids = d['input_ids'].to(device)
        print(input_ids.shape)
        lens.append(input_ids.shape[1])
        
        output_ids = model.generate(input_ids=input_ids,
                        do_sample=False,
                      temperature=0.9,
                      use_cache=True,
                      max_new_tokens=128,)
        
        output = tokenizer.decode(output_ids[0][len(input_ids[0]):],skip_special_tokens=True)
        if check_in_answers(output,answers):
            counts += 1
        print({"model_output":output,"answers":answers,"question":question})
        outs.append({"model_output":output,"answers":answers,"question":question,"idx":idx})
        
        
    print(lens)
    print(sum(lens)/len(lens))
    json.dump(outs,open(os.path.join(training_args.output_dir,str(data_args.end)+".json"),'w'))
    print("in rate:",counts/len(test_dataset))


if __name__ == "__main__":
    train()