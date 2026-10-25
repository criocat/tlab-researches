"""Small GPT-2 architecture from scratch: pre-LN, causal attention, learned
positions, GELU MLP, final LN, and tied item embeddings/output projection.
Only the last valid hidden state is projected to the full vocabulary.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.qkv = nn.Linear(dim, 3*dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(approximate="tanh"), nn.Linear(4*dim, dim))
        self.dropout = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x):
        b, t, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, d//self.heads).permute(2,0,3,1,4)
        q,k,v = qkv.unbind(0)
        a = F.scaled_dot_product_attention(q,k,v, is_causal=True, dropout_p=self.p if self.training else 0)
        x = x + self.dropout(self.proj(a.transpose(1,2).reshape(b,t,d)))
        return x + self.dropout(self.mlp(self.norm2(x)))


class GPT2Rec(nn.Module):
    def __init__(self, vocab, max_len=32, dim=64, layers=2, heads=4, dropout=.1):
        super().__init__()
        self.config = dict(vocab=vocab, max_len=max_len, dim=dim, layers=layers, heads=heads, dropout=dropout)
        self.item = nn.Embedding(vocab, dim, padding_idx=0)
        self.position = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim,heads,dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.apply(self._init)
        # GPT-2 residual projection initialization.
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=.02/math.sqrt(2*layers))
            nn.init.normal_(block.mlp[-1].weight, std=.02/math.sqrt(2*layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def hidden(self, tokens):
        # Right padding: a valid token cannot attend to later padding positions.
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.drop(self.item(tokens) + self.position(positions))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, tokens):
        h = self.hidden(tokens)
        end = tokens.ne(0).sum(1).clamp(min=1)-1
        last = h[torch.arange(len(tokens), device=tokens.device),end]
        return F.linear(last, self.item.weight)


class RewardModel(nn.Module):
    """History-aware reward regressor, independently initialized from policy.

    Trained only on observed interactions. Off-support scores are extrapolations.
    Outputs watch completion and a censored 3-day return proxy.
    """
    def __init__(self, vocab, dim=32):
        super().__init__()
        self.item = nn.Embedding(vocab, dim, padding_idx=0)
        self.net = nn.Sequential(nn.Linear(dim*4,64), nn.ReLU(), nn.Linear(64,2))
        self.bias = nn.Embedding(vocab,2)
        nn.init.zeros_(self.bias.weight)

    def forward(self, history, candidates):
        mask = history.ne(0).unsqueeze(-1)
        hist = self.item(history)
        pooled = (hist*mask).sum(1)/mask.sum(1).clamp(min=1)
        end = history.ne(0).sum(1).clamp(min=1)-1
        last = hist[torch.arange(len(history), device=history.device),end]
        item = self.item(candidates)
        if candidates.ndim == 2:
            pooled = pooled[:,None,:].expand_as(item)
            last = last[:,None,:].expand_as(item)
        z = torch.cat([pooled,last,item,pooled*item],dim=-1)
        return torch.sigmoid(self.net(z)+self.bias(candidates))


def append_tokens(history, new):
    """Append one generated item, truncate oldest context if full."""
    lengths = history.ne(0).sum(1)
    out = history.clone()
    full = lengths.eq(history.shape[1])
    out[full,:-1] = history[full,1:]
    out[full,-1] = new[full]
    rows = torch.arange(len(history),device=history.device)[~full]
    out[rows,lengths[~full]] = new[~full]
    return out
