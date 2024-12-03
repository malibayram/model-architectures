import torch
import torch.nn as nn
from typing import Tuple
from gemma_attention import GemmaAttention  # Custom self-attention module
from gemma_mlp import GemmaMLP  # Custom feedforward network module
from rms_norm import RMSNorm  # Custom normalization layer

class Gemma2DecoderLayer(nn.Module):
    """
    Implements a single decoder layer for the Gemma2 model.
    
    The decoder layer consists of:
    - Self-attention with rotary embeddings.
    - Layer normalization before and after self-attention.
    - An MLP block with optional pre and post feedforward layer normalization.
    """

    def __init__(
        self,
        self_attn: GemmaAttention,
        mlp: GemmaMLP,
        input_layernorm: RMSNorm,
        post_attention_layernorm: RMSNorm,
        pre_feedforward_layernorm: RMSNorm,
        post_feedforward_layernorm: RMSNorm,
    ):
        """
        Initializes the decoder layer with its components.

        Args:
            self_attn (GemmaAttention): Self-attention module.
            mlp (GemmaMLP): Multi-layer perceptron module for feedforward operations.
            input_layernorm (RMSNorm): Layer normalization before self-attention.
            post_attention_layernorm (RMSNorm): Layer normalization after self-attention.
            pre_feedforward_layernorm (RMSNorm): Layer normalization before MLP (optional).
            post_feedforward_layernorm (RMSNorm): Layer normalization after MLP (optional).
        """
        super().__init__()
        self.self_attn = self_attn
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.pre_feedforward_layernorm = pre_feedforward_layernorm
        self.post_feedforward_layernorm = post_feedforward_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the decoder layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            freqs_cis (torch.Tensor): Precomputed rotary embeddings.
            kv_write_indices (torch.Tensor): Indices for updating key-value cache.
            kv_cache (Tuple[torch.Tensor, torch.Tensor]): Tuple of cached keys and values.
            mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Output tensor of the same shape as `hidden_states`.
        """
        # Self-Attention Block
        residual = hidden_states  # Save input for residual connection
        hidden_states = self.input_layernorm(hidden_states)  # Normalize input
        hidden_states = self.self_attn(  # Apply self-attention
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)  # Post-attention normalization
        hidden_states = residual + hidden_states  # Add residual connection

        # MLP Block
        residual = hidden_states  # Save input for residual connection
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)  # Normalize before MLP
        hidden_states = self.mlp(hidden_states)  # Apply MLP
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)  # Normalize after MLP
        hidden_states = residual + hidden_states  # Add residual connection

        return hidden_states