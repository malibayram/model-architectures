import torch
import torch.nn as nn
from causal_self_attention import CausalSelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.attn_heads = nn.ModuleList([
            CausalSelfAttention(config, device) for _ in range(config.n_head)
        ])  # Create n_head attention heads
        self.projection = nn.Linear(config.n_embd * config.n_head, config.n_embd).to(device) # Linear layer to project multi-head attention outputs

    def forward(self, x):
        head_outputs = [head(x) for head in self.attn_heads] # Get the output of each attention head
        multihead_output = torch.cat(head_outputs, dim=-1) # Concatenate the outputs
        return self.projection(multihead_output) # Project the concatenated outputs

