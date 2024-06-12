# text_encoder.py

import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, n_heads):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads), 
            num_layers=n_layers
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return x
    
