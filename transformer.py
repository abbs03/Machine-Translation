import torch
import torch.nn as nn
from layers import *


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads:int,  d_ff:int ,dropout:int) -> None:
        super().__init__()

        self.mha = MultiHeadedAttention(embed_dim, n_heads, dropout)
        self.ffw = PositionWiseFeedForward(embed_dim, d_ff, dropout)
        
        self.l_n1 = nn.LayerNorm(embed_dim)
        self.l_n2 = nn.LayerNorm(embed_dim)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, enc_mask):
        # Self Attention
        x_norm = self.l_n1(x)
        x = x + self.dropout_1(self.mha(x_norm, x_norm, x_norm, enc_mask))
        
        # FeedForward
        x = x + self.dropout_2(self.ffw(self.l_n2(x)))
        return x



class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads:int,  d_ff:int ,dropout:int) -> None:
        super().__init__()
        
        self.self_mha = MultiHeadedAttention(embed_dim, n_heads, dropout)
        self.cross_mha = MultiHeadedAttention(embed_dim, n_heads, dropout)
        self.ffw = PositionWiseFeedForward(embed_dim, d_ff, dropout)
        
        self.l_n1 = nn.LayerNorm(embed_dim)
        self.l_n2 = nn.LayerNorm(embed_dim)
        self.l_n3 = nn.LayerNorm(embed_dim)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        
    def forward(self, x, dec_mask, enc_x, enc_mask):
        
        # Self Attention
        x_norm = self.l_n1(x)
        x = x + self.dropout_1(self.self_mha(x_norm, x_norm, x_norm, dec_mask))        
        
        # Cross Attention
        x = x + self.dropout_2(self.cross_mha(self.l_n2(x), enc_x, enc_x, enc_mask))
        
        # FeedForward
        x = x + self.dropout_3(self.ffw(self.l_n3(x)))
        return x
    

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, seq_len, embed_dim, n_heads, d_ff, n_blocks, dropout):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.src_tok_emb = TokenEmbeddding(src_vocab_size, embed_dim)
        self.tar_tok_emb = TokenEmbeddding(tgt_vocab_size, embed_dim)
        
        
        self.src_pos_emb = PositionEncoding(seq_len, embed_dim, dropout)
        self.tar_pos_emb = PositionEncoding(seq_len, embed_dim, dropout)
        
        
        self.encoder_blocks = nn.ModuleList([EncoderBlock(embed_dim, n_heads, d_ff, dropout) for _ in range(n_blocks)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(embed_dim, n_heads, d_ff, dropout) for _ in range(n_blocks)])
        
        self.projection = nn.Linear(embed_dim, tgt_vocab_size)
        
    def encoder(self, x, enc_mask):
        
        # (B x L) -> (B x L x C)
        x = self.src_tok_emb(x)
        x = self.src_pos_emb(x)
        
        for layer in self.encoder_blocks:
            x = layer(x, enc_mask)  #(B x L x C)
        
        return x
    
    def decoder(self, x, dec_mask, enc_x, enc_mask):
        
        # (B x L) -> (B x L x C)
        x = self.tar_tok_emb(x)
        x = self.tar_pos_emb(x)        
        
        for layer in self.decoder_blocks:
            x = layer(x, dec_mask, enc_x, enc_mask) #(B x L x C)
        
        # (B x L x C) -> (B x L x Tar_vocab)
        x = self.projection(x)
        
        return x
    
    def forward(self, enc_input, dec_input, enc_mask, dec_mask):
        
        self.enc_output = self.encoder(enc_input, enc_mask)
        logits = self.decoder(dec_input, dec_mask, self.enc_output, enc_mask)
        
        return logits
    
    
    def init_weights(self):
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            
        # Make the final projection layer less confident       
        self.projection.weight.data *= 0.1
        return     