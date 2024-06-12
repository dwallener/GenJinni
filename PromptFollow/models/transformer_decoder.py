# transformer_decoder.py

import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim, n_layers):
        super(TransformerDecoder, self).__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=n_heads),
            num_layers=n_layers
        )
        self.fc = nn.Linear(embed_dim, hidden_dim)
    
    def forward(self, tgt, memory):
        output = self.decoder(tgt, memory)
        output = self.fc(output)
        return output
    
