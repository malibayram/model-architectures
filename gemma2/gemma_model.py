import torch
import torch.nn as nn
from typing import List, Tuple
from gemma2_decoder_layer import Gemma2DecoderLayer
from rms_norm import RMSNorm
import gemma_config


class GemmaModel(nn.Module):

    def __init__(self, config: gemma_config.GemmaConfig, layers: List[Gemma2DecoderLayer]):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states