import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_len, d_model))
        
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        
        # Embedding + positional encoding
        x = self.embedding(input_ids) * (self.d_model ** 0.5)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Create attention mask for padding
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Convert to transformer format (True = attend, False = ignore)
        attn_mask = attention_mask == 0
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits
