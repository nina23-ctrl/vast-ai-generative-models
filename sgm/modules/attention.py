# SAFE-MATH ATTENTION
# Fully CPU-style math attention on CUDA
# No xformers, no SDPA, no flash, no triton, no kernels

import logging
import math
from inspect import isfunction
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

logpy = logging.getLogger(__name__)

# ==============================
# HARD DISABLE ALL BACKENDS
# ==============================
XFORMERS_DISABLED = True
XFORMERS_IS_AVAILABLE = False
SDP_IS_AVAILABLE = False

# ==============================
# HELPERS
# ==============================

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

# ==============================
# FEEDFORWARD
# ==============================

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)

# ==============================
# PURE MATH SELF ATTENTION
# ==============================

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b n (k h d) -> k b h n d", k=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.proj(out)

# ==============================
# PURE MATH CROSS ATTENTION
# ==============================

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if exists(mask):
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

# ==============================
# MEMORY SAFE CROSS ATTENTION
# ==============================

class MemoryEfficientCrossAttention(CrossAttention):
    pass  # math version == CrossAttention (safe fallback)

# ==============================
# BASIC TRANSFORMER BLOCK
# ==============================

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
    ):
        super().__init__()

        self.disable_self_attn = disable_self_attn

        self.attn1 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim if disable_self_attn else None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )

        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )

        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        if self.checkpoint:
            return checkpoint(self._forward, x, context)
        return self._forward(x, context)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

# ==============================
# SPATIAL TRANSFORMER
# ==============================

class SpatialTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
    ):
        super().__init__()

        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        elif context_dim is None:
            context_dim = [None] * depth

        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels, inner_dim, 1)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim[d],
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint,
            )
            for d in range(depth)
        ])

        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, 1))

    def forward(self, x, context=None):
        if not isinstance(context, list):
            context = [context]

        b, c, h, w = x.shape
        x_in = x

        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)
        return x + x_in

# ==============================
# LINEAR ATTENTION (SAFE)
# ==============================

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, hidden_dim, 1, bias=False)

        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, "b (h d) x y -> b h d (x y)", h=self.heads)
        k = rearrange(k, "b (h d) x y -> b h d (x y)", h=self.heads)
        v = rearrange(v, "b (h d) x y -> b h d (x y)", h=self.heads)

        k = k.softmax(dim=-1)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)

        out = rearrange(out, "b h d (x y) -> b (h d) x y", x=h, y=w)
        return self.to_out(out)
