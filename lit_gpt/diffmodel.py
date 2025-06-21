"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from flash_attn import flash_attn_func
from torch.nn.attention.flex_attention import flex_attention, create_block_mask #added
from functools import lru_cache #added
from lit_gpt.config import Config
from xformers.ops import SwiGLU
from .fused_rotary_embedding import apply_rotary_emb_func
RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")

# >>>> MODIFIED: compile flex_attention and cache masks
#flex_attention = torch.compile(flex_attention, dynamic=False)
torch._dynamo.config.cache_size_limit = 192 * 3
torch._dynamo.config.accumulated_cache_size_limit = 192 * 3

@lru_cache
def create_block_mask_cached(mask_mod, B: int, H: int, M: int, N: int, device="cuda", _compile=True):
    return create_block_mask(mask_mod, B, H, M, N, device=device, _compile=_compile)
# <<<< MODIFIED END

class TransEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size + 1, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.rope_cache: Optional[RoPECache] = None

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            # RWKV: set it to 1e-4
            # torch.nn.init.uniform_(module.weight,  -1e-4, 1e-4)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # GPT-NeoX
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (name == "w3.weight" and isinstance(module, SwiGLU) or (name=="proj.weight" and isinstance(module, SelfAttention))):  #if use xformer swiglu, fc2 layer will be renamed to w3
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd)  /  n_layer)


    def forward(
        self, idx: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor:
        B, T = idx.size()

        block_size = self.config.block_size
        assert block_size + self.config.latent_size >= T, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099

        cos, sin = self.rope_cache
        cos = cos[:T]
        sin = sin[:T]

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # >>>> MODIFIED: build block-wise attention mask
        attn_mask = None
        if self.config.attention_type == "block":
            def mask_mod(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
                group = self.config.block_size + self.config.latent_size
                return (kv_idx // group) <= (q_idx // group)

            attn_mask = create_block_mask_cached(
                mask_mod,
                B, self.config.n_head, T, T,
                device=x.device,
                _compile=True,
            )
        # <<<< MODIFIED END

        for i, block in enumerate(self.transformer.h):
            x = block(x, (cos, sin), attn_mask=attn_mask)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if return_hidden:
            return logits, x
        return logits  

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        seq_len = idx.size(1)
        return build_rope_cache(
            seq_len=seq_len,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device=idx.device,
            condense_ratio=self.config.condense_ratio,
        )


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = SelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.config = config
    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        attn_mask: Optional[torch.Tensor] = None,  # added
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:

        n_1 = self.norm_1(x)
        h = self.attn(n_1, rope,attn_mask=attn_mask)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = x + h + self.mlp(n_2)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )

            x = x + h
            x = x + self.mlp(self.norm_2(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        attn_mask: Optional[torch.Tensor] = None,  # added
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size) # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        # repeat k and v if necessary
        # Peiyuan: we do not need to do this as flash attention 2 already support GQA
        # if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
        #     # for MHA this is a no-op
        #     k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
        #     v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B,  T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.config.head_size)
        v = v.reshape(B,  T, -1, self.config.head_size)

        cos, sin = rope
        
        cos = cos.to(q.dtype) #added 
        sin = sin.to(q.dtype) #added

        # apply rope in fp32 significanly stabalize training
        # fused rope expect (batch_size, seqlen, nheads, headdim)
        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)

        # n_elem = int(self.config.rotary_percentage * self.config.head_size)

        # q_roped = apply_rope(q[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # k_roped = apply_rope(k[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # print( (q_roped - q).sum())
        # q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        # k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        y = self.scaled_dot_product_attention(q, k, v,attn_mask=attn_mask) #added

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,attn_mask: Optional[torch.Tensor] = None,  # added
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)
        
        # >>>> MODIFIED:
        if attn_mask is not None:
            return flex_attention(q,k,v,block_mask=attn_mask)
        # <<<< MODIFIED END

        if (
            FlashAttention2Available
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=False)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
             k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
             v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, scale=scale, is_causal=False
        )
        return y.transpose(1, 2)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        # self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        # self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        # self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=False, _pack_weights=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_fc_1 = self.fc_1(x)
        # x_fc_2 = self.fc_2(x)
        # x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        # return self.proj(x)
        return self.swiglu(x)


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)
