import math
import torch
import torch.nn as nn

class TokenEmbeddding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim:int):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.embed_dim)
    
    
class PositionEncoding(nn.Module):
    def __init__(self, block_size:int, embed_dim:int, dropout:float) -> None:
        super().__init__()
        
        # (LxC)
        pos_embed = torch.zeros(block_size, embed_dim)
        
        num = torch.arange(block_size).view(-1, 1)                                      # (Lx1)
        den = torch.exp( torch.arange(0, embed_dim, 2)/embed_dim * math.log(10_000))    # (C/2)
        
        # Fill up the Embedding
        pos_embed[:, 0::2] = torch.sin(num/den)     #(LxC/2)
        pos_embed[:, 1::2] = torch.cos(num/den)     #(LxC/2)
        
        pos_embed = pos_embed.view(1, block_size, embed_dim)
        self.register_buffer('pos_embed', pos_embed)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x - (B, L, C)
        x = x + self.pos_embed[:, :x.size(1),:]
        return self.dropout(x)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim:int, d_ff:int, dropout:float):
        super().__init__()
        
        
        self.fc_1 = nn.Linear(embed_dim, d_ff)
        self.fc_2 = nn.Linear(d_ff, embed_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        
        x = self.dropout(self.relu(self.fc_1(x)))
        x = self.fc_2(x)
        
        return x
        
        

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim:int, n_heads:int, dropout:float):
        super().__init__()
        
        assert embed_dim%n_heads == 0, 'embed_dim is not divisible by n_heads'
        
        self.n_heads = n_heads
        self.head_dim = embed_dim//n_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        
        B_q, L_q, C_q = q.shape
        B_k, L_k, C_k = k.shape
        
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        
        # (B x L x C) -> (B x L x n_heads x C//n_heads) -> (B x n_heads x L x Head_dim)
        q = q.view(B_q, L_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B_k, L_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B_k, L_k, self.n_heads, self.head_dim).transpose(1, 2)
        
        #Attention
        # (B x n_heads x L x head_dim) @ (B x n_heads x head_dim x L) -> (B x n_heads x L x L)
        affinity = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            affinity = affinity.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(affinity, dim = -1)
        attention = self.dropout(attention)
        
        # (B x n_heads x L x L) @ (B x n_heads x L x head_dim) -> (B x n_heads x L x head_dim)
        x = attention @ v

        #(B x n_heads x L x head_dim) -> (B x L x C)
        x = x.transpose(1, 2).reshape(B_q, L_q, C_q)
        
        return self.proj(x)
        

        
        