import os
os.environ["HF_DATASETS_CACHE"] = './data/.cache'
import random
import shutil
import gzip
import json
import torch
import requests
import datasets
import pickle
import transformers
from tqdm import tqdm
from datasets import Dataset
from collections import deque
from transformers import Trainer, HfArgumentParser, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pear_longma4train.ablation_40_LongMa import LongmaForCausalLM
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch.nn as nn
import numpy as np
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    logging,
    strtobool,
)

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)

if is_safetensors_available():
    import safetensors.torch

logger = logging.get_logger(__name__)
torch.set_printoptions(profile="full")


if os.path.exists('./data/.cache'):
    shutil.rmtree('./data/.cache')


class ChatDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        tokenizer = self.tokenizer
        pad_token_id = tokenizer.pad_token_id
        batch_input_ids = []
        batch_labels = []
        max_length = max([len(feature["input_ids"]) for feature in features])
        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            batch_input_ids.append([pad_token_id] * (max_length - len(input_ids)) + input_ids)
            batch_labels.append([pad_token_id] * (max_length - len(input_ids)) + labels)
        input_ids, labels = torch.LongTensor(batch_input_ids), torch.LongTensor(batch_labels)
        attention_mask = torch.where(input_ids == tokenizer.pad_token_id, 0, 1)
        labels[labels == tokenizer.pad_token_id] = -100
        position_ids = torch.cumsum(attention_mask, dim=-1) - 1
        position_ids[position_ids == -1] = 0
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "position_ids": position_ids}

class RandomMoverCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, features):
        tokenizer = self.tokenizer
        # pad_token_id = tokenizer.pad_token_id
        batch_input_ids = []
        batch_labels = []
        # max_length = max([len(feature["input_ids"]) for feature in features])
        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
        input_ids, labels = torch.LongTensor(batch_input_ids), torch.LongTensor(batch_labels)
        attention_mask = torch.where(input_ids == tokenizer.pad_token_id, 0, 1)
        labels[labels == tokenizer.pad_token_id] = -100
        # position_ids = torch.cumsum(attention_mask, dim=-1) - 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        dict2save = {n: p for n, p in self.model.state_dict().items() if 'alpha' in n}    
        torch.save(dict2save, os.path.join(output_dir, 'model.bin'))
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # neg_layers = [8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 26, 30]
        
        # for neg_layer in neg_layers:
        #     # breakpoint()
        #     loss += (torch.sum(self.model.model.layers[neg_layer].self_attn.alpha) - 32) ** 2
        return (loss, outputs) if return_outputs else loss

def print_example(example):
    print("Print Example of Training data:")
    print("*" * 20)
    print("*" * 20)
    print('Input ids:')
    print(tokenizer.batch_decode([example["input_ids"]]))
    print("*" * 20)
    print("Labels:")
    print(tokenizer.batch_decode([example["labels"]]))
    print('*' * 20)
    print('*' * 20)

def set_alphas(path, model, nm_layers):
    alphas = torch.load(path)
    nm_heads = {
                8: [22],
                10: [18],
                11: [6, 9],
                12: [10],
                13: [9, 14],
                14: [0, 15],
                15: [10, 14, 25],
                17: [0],
                18: [9],
                19: [15, 27],
                20: [26],
                26: [28],
                29: [15],
                30: [9]
            }

    for layeridx in nm_layers:
        for idx in nm_heads[layeridx]:
            alpha = alphas[f'model.layers.{layeridx}.self_attn.alpha']
            model.model.layers[layeridx].self_attn.alpha = nn.Parameter(alpha)
            print(f'({layeridx},{idx}):{model.model.layers[layeridx].self_attn.alpha[idx]}')

def set_alphas_5(path, model, nm_layers):
    alphas = torch.load(path)
    nm_heads = {
                26: [28],
                11: [6],
                14: [15],
                30: [9],
                18: [9]
            }
    
    for layeridx in nm_layers:
        for idx in nm_heads[layeridx]:
            alpha = alphas[f'model.layers.{layeridx}.self_attn.alpha']
            model.model.layers[layeridx].self_attn.alpha = nn.Parameter(alpha)
            print(f'({layeridx},{idx}):{model.model.layers[layeridx].self_attn.alpha[idx]}')
            
def set_alphas_10(path, model, nm_layers):
    alphas = torch.load(path)
    nm_heads = {
                26: [28],
                11: [6],
                14: [15],
                30: [9],
                18: [9],
                15: [10, 14],
                13: [9],
                12: [10],
                10: [18]
            }
    
    for layeridx in nm_layers:
        for idx in nm_heads[layeridx]:
            alpha = alphas[f'model.layers.{layeridx}.self_attn.alpha']
            model.model.layers[layeridx].self_attn.alpha = nn.Parameter(alpha)
            print(f'({layeridx},{idx}):{model.model.layers[layeridx].self_attn.alpha[idx]}')

def set_alphas_15(path, model, nm_layers):
    alphas = torch.load(path)
    nm_heads = {
                26: [28],
                11: [6],
                14: [0, 15],
                30: [9],
                18: [9],
                15: [10, 14, 25],
                13: [9],
                12: [10],
                10: [2, 18],
                19: [25],
                29: [15]
            }
    
    for layeridx in nm_layers:
        for idx in nm_heads[layeridx]:
            alpha = alphas[f'model.layers.{layeridx}.self_attn.alpha']
            model.model.layers[layeridx].self_attn.alpha = nn.Parameter(alpha)
            print(f'({layeridx},{idx}):{model.model.layers[layeridx].self_attn.alpha[idx]}')
            
def set_alphas_20(path, model, nm_layers):
    alphas = torch.load(path)
    nm_heads = {
                26: [28],
                11: [6],
                14: [0, 15],
                30: [9],
                18: [9],
                15: [10, 14, 25],
                13: [9],
                12: [10],
                10: [2, 18],
                19: [25],
                29: [15],
                31: [17],
                8: [22],
                17: [0],
                20: [26],
                9: [13]
            }
    
    for layeridx in nm_layers:
        for idx in nm_heads[layeridx]:
            alpha = alphas[f'model.layers.{layeridx}.self_attn.alpha']
            model.model.layers[layeridx].self_attn.alpha = nn.Parameter(alpha)
            print(f'({layeridx},{idx}):{model.model.layers[layeridx].self_attn.alpha[idx]}')

def set_alphas_25(path, model, nm_layers):
    alphas = torch.load(path)
    nm_heads = {
                26: [28],
                11: [6, 9],
                14: [0, 15],
                30: [9],
                18: [9],
                15: [10, 12, 14, 25],
                13: [9, 14],
                12: [10],
                10: [1, 2, 18],
                19: [25],
                29: [15],
                31: [17],
                8: [22],
                17: [0],
                20: [26],
                9: [13],
                7: [9]
            }
    
    for layeridx in nm_layers:
        for idx in nm_heads[layeridx]:
            alpha = alphas[f'model.layers.{layeridx}.self_attn.alpha']
            model.model.layers[layeridx].self_attn.alpha = nn.Parameter(alpha)
            print(f'({layeridx},{idx}):{model.model.layers[layeridx].self_attn.alpha[idx]}')

def set_alphas_30(path, model, nm_layers):
    alphas = torch.load(path)
    nm_heads = {
                26: [9, 28],
                11: [6, 9],
                14: [0, 15],
                30: [9],
                18: [9],
                15: [2, 7, 10, 12, 14, 25],
                13: [9, 14],
                12: [10],
                10: [1, 2, 18],
                19: [25],
                29: [15],
                31: [17],
                8: [22],
                17: [0],
                20: [26],
                9: [13, 16],
                7: [9],
                28: [22]
            }
    
    for layeridx in nm_layers:
        for idx in nm_heads[layeridx]:
            alpha = alphas[f'model.layers.{layeridx}.self_attn.alpha']
            model.model.layers[layeridx].self_attn.alpha = nn.Parameter(alpha)
            print(f'({layeridx},{idx}):{model.model.layers[layeridx].self_attn.alpha[idx]}')

def set_alphas_40(path, model, nm_layers):
    alphas = torch.load(path)
    nm_heads = {
                26: [9, 28, 19],
                11: [6, 9],
                14: [0, 15],
                30: [9],
                18: [9, 2],
                15: [2, 7, 10, 11, 12, 14, 25],
                13: [9, 14],
                12: [10, 13, 19],
                10: [1, 2, 3, 18],
                19: [25, 27],
                29: [15],
                31: [17],
                8: [22],
                17: [0],
                20: [26],
                9: [13, 16, 27, ],
                7: [9],
                28: [22],
                0: [25],
                16: [14]
            }
    
    for layeridx in nm_layers:
        for idx in nm_heads[layeridx]:
            alpha = alphas[f'model.layers.{layeridx}.self_attn.alpha']
            model.model.layers[layeridx].self_attn.alpha = nn.Parameter(alpha)
            print(f'({layeridx},{idx}):{model.model.layers[layeridx].self_attn.alpha[idx]}')

def load_dataset(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses()[0]
    training_args.remove_unused_columns = False
    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    n_gpus = torch.cuda.device_count()
    
    tokenizer = AutoTokenizer.from_pretrained('./meta-llama/Llama-2-7b-chat-hf', use_fast=False, padding_side="right")
    
    num_samples = 10000
    seq_len = 5
    train_data = load_dataset('./data/dataset_50000.pkl')
    train_dataset = Dataset.from_list(train_data)
    
    data_collator = ChatDataCollator(tokenizer)
    ckpt_dir = './meta-llama/Llama-2-7b-chat-hf'
    neg_layer_filter_5 = [26, 11, 14, 30, 18]
    neg_layer_filter_10 = [26, 11, 14, 30, 18, 15, 13, 12, 10]
    neg_layer_filter_15 = [26, 11, 14, 30, 18, 15, 13, 12, 10, 19, 29]
    neg_layer_filter_20 = [26, 11, 14, 30, 18, 15, 13, 12, 10, 19, 29, 31, 8, 17, 20, 9]
    neg_layer_filter_25 = [26, 11, 14, 30, 18, 15, 13, 12, 10, 19, 29, 31, 8, 17, 20, 9, 7]
    neg_layer_filter_30 = [26, 11, 14, 30, 18, 15, 13, 12, 10, 19, 29, 31, 8, 17, 20, 9, 7, 28]
    neg_layer_filter_40 = [26, 11, 14, 30, 18, 15, 13, 12, 10, 19, 29, 31, 8, 17, 20, 9, 7, 28, 0, 16]
    config = AutoConfig.from_pretrained(ckpt_dir)
    config._attn_implementation = 'eager'
    model = LongmaForCausalLM.from_pretrained(ckpt_dir, config=config, torch_dtype=torch.bfloat16)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    for neg_layer in neg_layer_filter_40:
        model.model.layers[neg_layer].self_attn.alpha = nn.Parameter(torch.ones(32, dtype=torch.bfloat16))

    for n, p in model.named_parameters():
        if 'alpha' in n:
            p: nn.Parameter.requires_grad = True
        else:
            p.requires_grad = False
    model.enable_input_require_grads()
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    if training_args.do_train:
        train_result = trainer.train()
        # breakpoint()
        trainer.save_model('./full_epoch_trained/40_50000_train_negfilter_epoch1')