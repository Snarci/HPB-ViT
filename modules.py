import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import math
import transformers
from transformers import AutoModelForImageClassification

"""
Size reduction for hierarchy reconstruction
"""
class HRecV2(nn.Module):
    def __init__(self, in_dim,embed_dim=768,inner_dim=2048):
        super().__init__()
        size_reduction=4
        self.embed_dim = embed_dim
        """I have a tensor of size [k,embed_dim] """
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, embed_dim//size_reduction),
            nn.LayerNorm(embed_dim//size_reduction)
        )
        self.mpl_embed = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, in_dim),
            nn.LayerNorm(in_dim)
        )

    def forward(self, x):
         
        x = self.mlp(x)
        b, n, c = x.shape
        size_side_patch = int(math.sqrt(c))
        side_size_cube = int(math.sqrt(n))
        side_size_embedding = int(math.sqrt(self.embed_dim))
        x = Rearrange('b (nh nw) (h w) -> b (nh h) (nw w)', nh=side_size_cube, nw=side_size_cube, h=size_side_patch, w=size_side_patch)(x)
        x = Rearrange('b (h p1) (w p2) -> b (h w) (p1 p2)', p1=side_size_embedding, p2=side_size_embedding)(x)
        x = self.mpl_embed(x)
        return x

    
"""
Patch merger from the paper
"""
class PatchMerger(nn.Module):
    def __init__(self, dim, num_tokens_out):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))

    def forward(self, x):
        x = self.norm(x)
        sim = torch.matmul(self.queries, x.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim = -1)
        return torch.matmul(attn, x)
"""
Sum and normalization
"""  
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
"""
Feed forward layer
"""
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
"""
Attention layer from the paper
"""
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
"""
Transformer layer
"""
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

