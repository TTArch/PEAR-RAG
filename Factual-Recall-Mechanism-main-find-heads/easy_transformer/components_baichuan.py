# flake8: noqa: B950
import math
# from mimetypes import init
from typing import Callable, Union, List, Tuple, Dict, Optional
import torch.utils.checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import einops
import warnings
import logging
from numpy import dtype

# from functools import *

from easy_transformer.hook_points import HookPoint
from easy_transformer.utils import gelu_new, solu, gelu_fast
from easy_transformer.EasyTransformerConfig import EasyTransformerConfig

# from fancy_einsum import einsum

from easy_transformer.caching import (
    EasyTransformerKeyValueCache,
    EasyTransformerKeyValueCacheEntry,
)

import math
import warnings
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch.utils.checkpoint
from numpy import dtype
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.versions import require_version

# from .configuration_milm import MiConfig

warnings.filterwarnings("ignore")

logger = logging.get_logger(__name__)

# var = []

def get_torch_dtype(config):
    dtype = torch.float32
    # dtype = torch.get_default_dtype()
    # if dtype == torch.float32:
    # if config.torch_dtype == "float16":
    #     dtype = torch.float16
    # elif config.torch_dtype == "bfloat16":
    #     dtype = torch.bfloat16
    return dtype


def _yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return _get_interleave_power_of_2(closest_power_of_2) + \
               _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

def _gen_alibi_mask(n_head, max_pos):
    """used in inference only"""
    slopes = torch.Tensor(_get_interleave(n_head))
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_pos).unsqueeze(0).unsqueeze(0).expand(
        n_head, -1, -1)
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(
        _fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1
    )
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask

global_future_mask = None

# def initialize_future_mask(n_heads, max_pos):
#     if future_mask is None:
#         register_buffer("future_mask", _gen_alibi_mask(self.cfg.n_heads, self.max_pos), persistent=False)


def initialize_future_mask(n_heads, max_pos):
    global global_future_mask
    if global_future_mask is None:
        global_future_mask = _gen_alibi_mask(n_heads, max_pos)
# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        # self.W_E = nn.Parameter(torch.empty(self.cfg.d_vocab, self.cfg.d_model, dtype=get_torch_dtype(cfg)))
        self.embed_tokens = nn.Embedding(
            self.cfg.d_vocab,
            self.cfg.d_model,
            self.cfg.padding_idx,
            dtype=get_torch_dtype(cfg),
        )

    def forward(self, tokens):
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        embed = self.embed_tokens(tokens)
        return embed# Shape [batch pos d_model]


class Unembed(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):

        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.output_projection = nn.Linear(
            self.cfg.d_model,
            self.cfg.d_vocab,
            bias=False,
            dtype=get_torch_dtype(cfg),
        )

    def forward(self, residual):
        return self.output_projection(residual)






# LayerNormPre
# I fold the LayerNorm weights and biases into later weights and biases.
# This is just the 'center and normalise' part of LayerNorm
# Centering is equivalent to just deleting one direction of residual space,
# and is equivalent to centering the weight matrices of everything writing to the residual stream
# Normalising is a funkier non-linear operation, that projects the residual stream onto the unit hypersphere






class RMSNorm(nn.Module):
    def __init__(
        self, cfg: Union[Dict, EasyTransformerConfig], length: Optional[int] = None, eps=1e-6
    ):

        """
        RMSNorm - LayerNorm without the centering and bias (RMS = Root Mean Square)

        length (Optional[int]): If the dimension of the RMSNorm. If not provided, assumed to be d_model
        """
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps
        self.length = self.cfg.d_model


        self.w = nn.Parameter(torch.ones(self.length))

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        self.hook_normalized = HookPoint()  # [batch, pos, length]
    def forward(self, x):
        input_dtype = x.dtype
        # from easy_transformer.modeling_milm import PRE
        device = x.device
        # x = PRE.pop(0)
        x = x.to(torch.float32)
        # variance = x.detach().to(torch.float32).pow(2).cpu().mean(-1, keepdim=True).to(device)
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # var.append(variance)
        variance = self.hook_scale(variance)
        # x = self.hook_normalized(x.detach() * torch.rsqrt((variance + self.eps).cpu()).to(device))
        x = self.hook_normalized(x * torch.rsqrt(variance + self.eps).to(device))
        out = self.w * x.to(input_dtype)
        return out

# class RMSNorm(torch.nn.Module):
#     def __init__(self, hidden_size, epsilon=1e-6):
#         super().__init__()
#         self.weight = torch.nn.Parameter(torch.empty(hidden_size))
#         self.epsilon = epsilon

#     def forward(self, hidden_states):
#         variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)

#         # convert into half-precision
#         if self.weight.dtype in [torch.float16, torch.bfloat16]:
#             hidden_states = hidden_states.to(self.weight.dtype)

#         return self.weight * hidden_states


# Attention
class Attention(nn.Module):
    def __init__(
        self, cfg: Union[Dict, EasyTransformerConfig], attn_type="global", layer_id=None
    ):
        """Attention Block - params have shape [head_index, d_model, d_head] (or [head_index, d_head, d_model] for W_O) and multiply on the right. attn_scores refers to query key dot product immediately before attention softmax

        Convention: All attention pattern-style matrices have shape [batch, head_index, query_pos, key_pos]

        Args:
            cfg (Union[Dict, EasyTransformerConfig]): Config
            attn_type (str, optional): "global" or "local", used by GPT-Neo. Local attention means the model can only attend back cfg.window_size tokens (here, 256). Not used by any other model at the moment. Defaults to "global".
            layer_id (int, optional): The index of the current layer. Used by the Mistal models (labelled here as stanford-gpt2) to scale down attention scores pre softmax for numerical stability reasons by 1/(layer_id+1). Defaults to None.
        """
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.q_proj = nn.Linear(
            self.cfg.d_model,
            self.cfg.n_heads * self.cfg.d_head,
            bias=False,
            dtype=get_torch_dtype(cfg),
        )
        self.k_proj = nn.Linear(
            self.cfg.d_model,
            self.cfg.n_heads * self.cfg.d_head,
            bias=False,
            dtype=get_torch_dtype(cfg),
        )
        self.v_proj = nn.Linear(
            self.cfg.d_model,
            self.cfg.n_heads * self.cfg.d_head,
            bias=False,
            dtype=get_torch_dtype(cfg),
        )
        self.out_proj = nn.Linear(
            self.cfg.n_heads * self.cfg.d_head,
            self.cfg.d_model,
            bias=False,
            dtype=get_torch_dtype(cfg),
        )

        self.attn_type = attn_type

        self.dtype = get_torch_dtype(cfg)


        self.layer_id = layer_id
        self.attn_scale = math.sqrt(self.cfg.d_head)


        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]

        # Added by Arthur; all are ResidPre, but we want finer access
        # assert self.cfg.positional_embedding_type in [
        #     "standard",
        #     "rotary",
        # ], f"q_input hooks only support standard and rotary positional embeddings, not {self.cfg.positional_embedding_type}"
        # self.hook_q_input = HookPoint()  # [batch, pos, d_model]
        # self.hook_k_input = HookPoint()  # [batch, pos, d_model]
        # self.hook_v_input = HookPoint()  # [batch, pos, d_model]

        self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_attn = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_result = HookPoint()  # [batch, query_pos, head_index, d_model]

        # See EasyTransformerConfig for more details.
        # if self.cfg.positional_embedding_type == "shortformer":
        #     # This tracks the input to the keys and queries, which is resid_pre + pos_embeds
        #     self.hook_attn_input = HookPoint()  # [batch, pos, d_model]
        # elif self.cfg.positional_embedding_type == "rotary":
        #     # Applies a rotation to each two-element chunk of keys and queries pre dot producting to bake in relative position. See EasyTransformerConfig for details
        self.hook_rot_k = HookPoint()
        self.hook_rot_q = HookPoint()
        # sin, cos = self.calculate_sin_cos_rotary(
        #     self.cfg.rotary_dim, self.cfg.n_ctx
        # )
        
        self.alibi_mask = None
        self.max_pos = cfg.n_ctx  # Maximum sequence length
        # self.register_buffer("rotary_sin", sin)
        # self.register_buffer("rotary_cos", cos)
        
        initialize_future_mask(self.cfg.n_heads, self.cfg.n_ctx)

    def compute_alibi_mask(self, batch_size, n_heads, q_len, kv_len, device):
        # Slice the alibi_mask to match q_len and kv_len
        # alibi_mask = global_future_mask[:, :q_len, :kv_len]
        alibi_mask = global_future_mask[:, :q_len, :kv_len]
        # Expand to batch size
        alibi_mask = alibi_mask.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
        return alibi_mask  # Shape: [batch_size, n_heads, q_len, kv_len]

    def forward(
        self,
        resid_pre: torch.Tensor,  # goddamn normalized thing
        shortformer_pos_embed: Optional[torch.Tensor] = None,
        past_kv_cache_entry: Optional[EasyTransformerKeyValueCacheEntry] = None,
        no_reduce: Optional[bool] = False,
    ):
        """
        shortformer_pos_embed is only used if self.cfg.positional_embedding_type == "shortformer", else defaults to None and is irrelevant. See EasyTransformerConfig for more details
        past_kv_cache_entry is an optional entry of past keys and values for this layer, only relevant if generating text. Defaults to None
        """
        # device = resid_pre.device
        # from easy_transformer.modeling_milm import PRE
        # resid_pre = PRE.pop(0)
        # resid_pre = resid_pre.to(device)
        # print(resid_pre)
        # input()
        # if self.cfg.use_headwise_qkv_input:
        #     assert self.cfg.positional_embedding_type in ["standard", "rotary"]
        #     warnings.warn("Using the new way of doing qkv input")
        # head_input = einops.repeat(
        #     resid_pre, "a b c -> a b x c", x=self.cfg.n_heads
        # )
        bsz, q_len, d_model = resid_pre.size()
        # var.append(resid_pre)
        q = self.q_proj(resid_pre.detach().clone())
        k = self.k_proj(resid_pre.detach().clone())
        v = self.v_proj(resid_pre.detach().clone())

        # var.append(k)
        # var.append(v)

        q = q.view(bsz, q_len, self.cfg.n_heads, self.cfg.d_head)
        k = k.view(bsz, q_len, self.cfg.n_heads, self.cfg.d_head)
        v = v.view(bsz, q_len, self.cfg.n_heads, self.cfg.d_head)
        # print(q.transpose(1, 2))
        # input()

        q = self.hook_q(q)
        k = self.hook_k(k)
        v = self.hook_v(v)

        
        if past_kv_cache_entry is not None:
            # assert past_kv_cache_entry is None, "past_kv_cache_entry is not None"
            # Appends the new keys and values to the cached values, and automatically updates the cache
            # kv_cache_pos_offset = past_kv_cache_entry.past_keys.size(1)
            k, v = past_kv_cache_entry.append(k, v)
            kv_len = k.size(1)
        else:
            # Not using a cache
            # kv_cache_pos_offset = 0
            kv_len = k.size(1)

        q = q.transpose(1, 2) #[bsz, head, q_len, kv_len]

        if past_kv_cache_entry is not None:
            k, v = past_kv_cache_entry.append(k, v)
            k = k.to(q)
            v = v.to(q)
        
        k = self.hook_rot_k(k)
        k = k.transpose(1, 2) #[bsz, head, q_len, kv_len]
        v = v.transpose(1, 2) #[bsz, head, q_len, kv_len]
        
        attn_scores = torch.matmul(
                q, k.transpose(2, 3))/ math.sqrt(self.cfg.d_head)
        

         # [batch, head_index, query_pos, key_pos]
        attn_scores = torch.clamp(attn_scores, min=-1024.0, max=1024.0)
        alibi_mask = self.compute_alibi_mask(bsz, self.cfg.n_heads, q_len, kv_len, q.device)
        attn_scores = attn_scores + alibi_mask
        # if self.cfg.attention_dir == "causal":
        #     # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
        # attn_scores = self.apply_causal_mask(
        #     attn_scores, kv_cache_pos_offset
        # )  # [batch, head_index, query_pos, key_pos]
        
        attn_scores = self.hook_attn_scores(attn_scores)
        
        attn_matrix = self.hook_attn(
            nn.functional.softmax(
                attn_scores, dim=-1, dtype=torch.float32
            )
        )  # [batch, head_index, query_pos, key_pos]
        attn_matrix = attn_matrix.to(q.dtype)
        
        # input()
        z = torch.matmul(attn_matrix, v)
        z = z.transpose(1, 2).contiguous() #"batch pos head_index d_head
        z = self.hook_z(z)
        
        # z = z.reshape(bsz, q_len,  d_model)
        
        # print(z)
        # input()
        # out = self.out_proj(z)
        # print(out)
        wo = self.out_proj.weight.reshape(d_model, -1, self.cfg.d_head)
        # print(z.shape)
        out = torch.einsum(" ...ikj,...kj ->...ki",[wo, z])
        out = self.hook_result(out) # [bsz, q_len, head_idx, d_model]
        out = out.sum(-2) # [bsz, q_len, d_model]
        # breakpoint()
        # input()
        return out

    def apply_causal_mask(self, attn_scores, past_kv_pos_offset):
        # The query context length is the number of positions we take queries from - if not using a past_kv_cache this is just the context length (for the current prompt), but if we're caching it's just a single token.
        # query_ctx_length = attn_scores.size(-2)
        # The key context length is the number of positions in the past - this includes all positions in the cache
        # If not caching, query_ctx_length == key_ctx_length
        # key_ctx_length = attn_scores.size(-1)
        bsz, head_num, query_ctx_length, key_ctx_length = attn_scores.size()

        assert (
            query_ctx_length + past_kv_pos_offset == key_ctx_length
        ), f"query_ctx_length {query_ctx_length} + past_kv_pos_offset {past_kv_pos_offset} != key_ctx_length {key_ctx_length} - you likely have a bug."
        mask = torch.full(
            (self.cfg.n_ctx, self.cfg.n_ctx),
            torch.tensor(torch.finfo(self.dtype).min),
        )
        mask_cond = torch.arange(mask.size(-1))
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(self.dtype)

        mask = mask[past_kv_pos_offset : past_kv_pos_offset + query_ctx_length, :key_ctx_length]
        mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, query_ctx_length, key_ctx_length)
        return attn_scores + mask.to(attn_scores)
        # return torch.where(
        #     self.mask[
        #         past_kv_pos_offset : past_kv_pos_offset + query_ctx_length,
        #         :key_ctx_length,
        #     ],
        #     attn_scores,
        #     0,
        # )

    def shortformer_calculate_qk(self, x, shortformer_pos_embed):
        # We add on the positional encodings to the residual stream JUST for the keys and queries, it's not added to the normal residual stream.
        attn_input = self.hook_attn_input(
            x + shortformer_pos_embed
        )  # [batch, pos, d_model]
        q = self.hook_q(
            einsum(
                "batch pos d_model, head_index d_model d_head \
                -> batch pos head_index d_head",
                attn_input,
                self.W_Q,
            )
            + self.b_Q
        )  # [batch, pos, head_index, d_head]
        k = self.hook_k(
            einsum(
                "batch pos d_model, head_index d_model d_head \
                -> batch pos head_index d_head",
                attn_input,
                self.W_K,
            )
            + self.b_K
        )  # [batch, pos, head_index, d_head]
        return (q, k)

    def rotary_rotate_qk(self, q, k, past_kv_pos_offset):
        # We first apply standard q and k calculation

        q = self.hook_rot_q(self.apply_rotary(q, past_kv_pos_offset))
        k = self.apply_rotary(k, past_kv_pos_offset)
        return q, k
    
    # MiLM-32k
    def calculate_sin_cos_rotary(self, rotary_dim, n_ctx, base=10000):
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim))
        t = torch.arange(n_ctx)
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        angles = torch.cat((freqs, freqs), dim=-1) #[pos, dim]

        return torch.sin(angles), torch.cos(angles)
    
    # def calculate_sin_cos_rotary(self, rotary_dim, n_ctx, base=10000):
    #     """
    #     Calculate the sine and cosine waves to use in a rotary embedding. See https://blog.eleuther.ai/rotary-embeddings/ for details
    #     """
    #     pos = torch.arange(n_ctx, dtype=torch.float32)
    #     dim = torch.arange(rotary_dim // 2, dtype=torch.float32)
    #     # A set of frequencies evenly spaced in log space
    #     freq = base ** (dim / (rotary_dim / 2))
    #     freq = einops.repeat(freq, "d -> (d 2)")
    #     # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
    #     angles = pos[:, None] / freq[None, :]
    #     return torch.sin(angles), torch.cos(angles)

    # def rotate_every_two(self, x):
    #     """
    #     Rotary helper function, splits x into blocks of size 2 along the final axis and maps [x0, x1] to [-x1, x0]
    #     """
    #     rot_x = x.clone()
    #     rot_x[..., 0::2] = -x[..., 1::2]
    #     rot_x[..., 1::2] = x[..., 0::2]
    #     return rot_x

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(self, x, past_kv_pos_offset=0):
        # Only apply rotary to first rotary_dim dimensions (eg, if rotary_dim=64 and d_head=256, only apply to first 1/4 of dimensions)
        x_pos = x.size(2)
        sin = self.rotary_sin[past_kv_pos_offset : past_kv_pos_offset + x_pos, :].unsqueeze(0).unsqueeze(1).to(x)
        cos = self.rotary_cos[past_kv_pos_offset : past_kv_pos_offset + x_pos, :].unsqueeze(0).unsqueeze(1).to(x)
        embed = (x * cos) + (self.rotate_half(x) * sin)
        return embed


def get_act(act):
    ACT2CLS = {
        "leaky_relu": nn.LeakyReLU,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "sigmoid": nn.Sigmoid,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "tanh": nn.Tanh,
    }
    act_args = ACT2CLS[act] 
    if isinstance(act_args, tuple):
        return act_args[0](act_args[1])
    else:
        return act_args({})

class MLP(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        super().__init__()
        dim = cfg.d_model
        intermediate_size = cfg.ffn_dim_multiplier

        self.w1 = nn.Linear(
            dim,
            intermediate_size,
            bias=False,
            dtype=get_torch_dtype(cfg),
        )
        self.w2 = nn.Linear(
            dim,
            intermediate_size,
            bias=False,
            dtype=get_torch_dtype(cfg),
        )
        self.w3 = nn.Linear(
            intermediate_size,
            dim,
            bias=False,
            dtype=get_torch_dtype(cfg),
        )
        self.act_fn = get_act(cfg.act_fn)
        # self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        # # self.hook_mid = HookPoint()
        # self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        # self.all_reduce = lambda x: x

    def forward(self, x):
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        
        # pre_act = self.hook_pre(
        #     self.w1(x)
        # )  # [batch, pos, d_mlp]
        # # if not self.cfg.act_fn.endswith("_ln"):
        # #     post_act = self.hook_post(F.silu(pre_act))  # [batch, pos, d_mlp]
        # # else:
        # mid_act = F.silu(pre_act)  # [batch, pos, d_mlp]
        
        # post_act = self.hook_post(self.w2(x) * mid_act)
        
        # mlp_out = (
        #     self.w3(post_act)
        # )  # [batch, pos, d_model]
        x1 = self.w1(x)
        x2 = self.w2(x)
        mlp_out = self.w3(self.act_fn(x1) * x2) 

        # var.append(x)
        # var.append(x1)
        # var.append(x2)
        # var.append(mlp_out)
        return mlp_out

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig], block_index):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.ln1 = RMSNorm(cfg)
        self.ln2 = RMSNorm(cfg)
        # if self.cfg.normalization_type == "LN":
        #      #RMSNorm
        #     if not self.cfg.attn_only:
        #          #RMSNorm
        # elif self.cfg.normalization_type == "LNPre":
        #     # We've folded in LayerNorm weights, so just need the center + scale parts
        #     warnings.warn("Moved LN1 to the attention block")
        #     if not self.cfg.attn_only:
        #         self.ln2 = RMSNormPre(cfg)
        # elif self.cfg.normalization_type is None:
        #     self.ln1 = nn.Identity()
        #     if not self.cfg.attn_only:
        #         self.ln2 = nn.Identity()
        # else:
        #     logging.warning(
        #         f"Invalid normalization_type passed in {self.cfg.normalization_type}"
        #     )

        # if not self.cfg.use_local_attn:
        self.attn = Attention(cfg, "global", block_index)
        # else:
        #     assert self.cfg.attn_types is not None
        #     attn_type = self.cfg.attn_types[block_index]
        #     self.attn = Attention(cfg, attn_type, block_index)
        # if not self.cfg.attn_only:
        self.mlp = MLP(cfg)

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        if not self.cfg.attn_only and not self.cfg.parallel_attn_mlp:
            self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(
        self,
        resid_pre: torch.Tensor,
        shortformer_pos_embed: Optional[torch.Tensor] = None,
        past_kv_cache_entry: Optional[EasyTransformerKeyValueCacheEntry] = None,
    ):
        """A single Transformer block.

        Args:
            resid_pre (torch.Tensor): The residual stream - shape [batch, pos, d_model]
            cache (EasyTransformerKeyValueCache): A cache of previous keys and values, used only when generating text. Defaults to None.
            shortformer_pos_embed (torch.Tensor, optional): Only used for positional_embeddings_type == "shortformer". The positional embeddings. See EasyTransformerConfig for details. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        resid_pre = self.hook_resid_pre(resid_pre)  # [batch, pos, d_model]
        # normalized_resid_pre = self.ln1(resid_pre)
        norm_resid_pre = self.ln1(resid_pre)

        # print(resid_pre)
        # input()
        attn_out = self.hook_attn_out(
            self.attn(
                norm_resid_pre,  # edited by Arthur from normalized ... so we can go headwise
                shortformer_pos_embed=shortformer_pos_embed,
                past_kv_cache_entry=past_kv_cache_entry,
            )
        )  # [batch, pos, d_model]
        
        # if not self.cfg.attn_only and not self.cfg.parallel_attn_mlp:
        # print(attn_out)
        # input()
        resid_mid = self.hook_resid_mid(
            resid_pre + attn_out
        )  # [batch, pos, d_model]


        normalized_resid_mid = self.ln2(resid_mid)

        mlp_out = self.hook_mlp_out(
            self.mlp(normalized_resid_mid)
        )  # [batch, pos, d_model]

        resid_post = self.hook_resid_post(
            resid_mid + mlp_out
        )
        # var.append(normalized_resid_mid)
        # var.append(mlp_out)
        # var.append(resid_post)
        # [batch, pos, d_model]
        # elif self.cfg.parallel_attn_mlp:
        #     # Dumb thing done by GPT-J, both MLP and Attn read from resid_pre and write to resid_post, no resid_mid used.
        #     # In GPT-J, LN1 and LN2 are tied, in GPT-NeoX they aren't.
            
        #     normalized_resid_pre_2 = self.ln2(resid_pre)
        #     mlp_out = self.hook_mlp_out(
        #         self.mlp(normalized_resid_pre_2)
        #     )  # [batch, pos, d_model]
        #     resid_post = self.hook_resid_post(
        #         resid_pre + attn_out + mlp_out
        #     )  # [batch, pos, d_model]
        # else:
        #     resid_post = self.hook_resid_post(
        #         resid_pre + attn_out
        #     )  # [batch, pos, d_model]

        
        return resid_post
