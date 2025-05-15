import math

import torch
import torch.nn as nn
from gpt_block import GPTBlock


def get_position_encoding(seq_len:int, d:int, n:int=10000, device:str='cpu'):
    """
    Computes the positional encoding matrix of shape (seq_len, d).
    
    Args:
        seq_len (int): Length of the sequence.
        d (int): Dimension of the embedding.
        n (float): The base for the exponential term (default 10000 in many Transformer implementations).
        device (str): The device to place the tensor on ('cpu' or 'cuda').
    
    Returns:
        torch.Tensor: A tensor of shape (seq_len, d) containing the positional encodings.
    """
    
    P = torch.zeros(seq_len, d, device=device)
    for pos in range(seq_len):
        for i in range(0, d // 2):
            P[pos, 2 * i] = math.sin(pos / (n ** ((2 * i) / d)))
            if i + 1 < d:
                P[pos, 2 * i + 1] = math.cos(pos / (n ** ((2 * i) / d)))

    return P.unsqueeze(0)

class GPTModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd).to(device)
        self.position_encoding = get_position_encoding(config.seq_len, config.n_embd, n=10000, device=device)
        self.blocks = nn.Sequential(*[GPTBlock(config, device) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd).to(device)
        self.head = nn.Linear(config.n_embd, config.vocab_size).to(device)
    
    def forward(self, x):
        x = self.token_embedding(x) + self.position_encoding
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)