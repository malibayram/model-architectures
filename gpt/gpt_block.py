import torch.nn as nn
from multi_head_attention import MultiHeadAttention


class GPTBlock(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.mha = MultiHeadAttention(config, device)
        self.ln1 = nn.LayerNorm(config.n_embd).to(device)
        """ self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        ).to(device)
        self.ln2 = nn.LayerNorm(config.n_embd).to(device) """

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        # x = x + self.ffn(self.ln2(x))
        return x