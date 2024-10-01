import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3,4,5'
# CUDA_VISIBLE_DEVICES = '1,3,4,5'
import torch
print(torch.cuda.device_count())
import plotly.express as px
results = torch.zeros(size=(1, 2))
fig = px.imshow(
    results,
    title="",
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
)

# #os.environ['HF_HOME'] = "/baichuan/huggingface_model"

import torch
# assert torch.cuda.device_count() == 1
from tqdm import tqdm
import pandas as pd
import torch
import torch as t
from easy_transformer.EasyTransformer_baichuan import (
    EasyTransformer,
)
from time import ctime
from functools import partial

import numpy as np
from tqdm import tqdm
import pandas as pd

from easy_transformer.experiment import (
    ExperimentMetric,
    AblationConfig,
    EasyAblation,
    EasyPatching,
    PatchingConfig,
)
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import random
import einops
# from IPython import get_ipython
from copy import deepcopy
from easy_transformer.fact_dataset_llama import (
    FactDataset,
)
from easy_transformer.fact_utils import (
    path_patching,
    max_2d,
    show_pp,
    show_attention_patterns,
    plot_path_patching,
    scatter_attention_and_contribution,
)
from random import randint as ri
from easy_transformer.fact_circuit_extraction import (
    do_circuit_extraction,
    get_heads_circuit,
    get_mlps_circuit
)
from easy_transformer.fact_utils import logit_diff, probs
from transformers import AutoModelForCausalLM, AutoTokenizer
# from easy_transformer.ioi_utils import get_top_tokens_and_probs as g

# ipython = get_ipython()
# if ipython is not None:
#     ipython.magic("load_ext autoreload")
#     ipython.magic("autoreload 2")
class FactDataset_copy:
    def __init__(
        self,
        N=10,
        seq_len=500,
        tokenizer=None,
        prompts=None,
        symmetric=False,
        prefixes=None,
        nb_templates=None,
        ioi_prompts_for_word_idxs=None,
        prepend_bos=False,
        manual_word_idx=None,
        counterfact=False,
        nation=None,
        add_prefix=0,
    ):
        """
        ioi_prompts_for_word_idxs:
            if you want to use a different set of prompts to get the word indices, you can pass it here
            (example use case: making a ABCA dataset)
        """
        self.N = N
        starts = torch.zeros(N, dtype=torch.int)
        end = torch.full((N,), 2 * seq_len - 2, dtype=torch.int)

        self.word_idx = {'starts': starts, 'end': end}

        input_ids = torch.randint(0, len(tokenizer), (N, seq_len))
        self.toks = input_ids.repeat(1, 2)
        self.IW_tokenIDs = self.toks[:, seq_len - 1]
    def __len__(self):
        return self.N

seed = 10024
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

model = EasyTransformer.from_pretrained("Baichuan-13B-Chat").cuda()

model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
experment_list = [(5, 80), (10, 50), (30, 20), (60, 10)]
tokenizer = AutoTokenizer.from_pretrained("/data/tantaoruc/tantao/public/Baichuan-13B-Chat", use_fast=False, trust_remote_code=True)

for N, seq_len in experment_list:
    # if index <= 4:
    #     continue
    figure_save_dir = f'output_figures/test_baichuan_ABAB_0918test/{N}/{model.cfg.model_name}'
    os.makedirs(figure_save_dir, exist_ok=True)

    tensor_save_dir = f'saved_tensors/test_baichuan_ABAB_0918test/{N}/{model.cfg.model_name}'
    os.makedirs(tensor_save_dir, exist_ok=True)

    f_dataset = FactDataset_copy(
        N=N,
        seq_len=seq_len,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        counterfact=False,
    )

    with torch.no_grad():
        plot_path_patching(
            model,
            D_new=f_dataset,
            D_orig=f_dataset,
            receiver_hooks=[(f"blocks.{model.cfg.n_layers-1}.hook_resid_post", None)],
            position="end",
            figure_save_dir=figure_save_dir,
            layout='max+1',
            title='logits',
            threshold=1,
            metric='iw',
        )

    model.reset_hooks()
