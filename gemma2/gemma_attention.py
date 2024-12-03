import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from linear import Linear  # Custom Linear layer assumed to be defined in `linear.py`
from gemma_functions import apply_rotary_emb  # Function to apply rotary embeddings
import gemma_config  # Configuration for GemmaAttention, including AttentionType

class GemmaAttention(nn.Module):
    """
    Implements multi-head attention with support for rotary embeddings, sliding window masking,
    and key-value caching.

    The class handles:
    - Query, Key, Value (QKV) projection.
    - Applying rotary positional embeddings.
    - Key-value caching for efficient sequential processing.
    - Scaled dot-product attention with optional sliding window masking.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        attn_logit_softcapping: Optional[float],
        query_pre_attn_scalar: Optional[int],
        head_dim: int,
        attn_type: gemma_config.AttentionType,
        qkv_proj: Linear,
        o_proj: Linear,
        sliding_window_size: Optional[int] = None,
    ):
        """
        Initializes the GemmaAttention module.

        Args:
            hidden_size (int): Input tensor dimension.
            num_heads (int): Number of attention heads.
            num_kv_heads (int): Number of key-value attention heads.
            attn_logit_softcapping (Optional[float]): Softcapping value for attention logits.
            query_pre_attn_scalar (Optional[int]): Pre-scaling factor for queries.
            head_dim (int): Dimension of each attention head.
            attn_type (gemma_config.AttentionType): Type of attention (e.g., local sliding).
            sliding_window_size (Optional[int]): Size of the sliding window for local attention.
        """
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        # Ensure num_heads is divisible by num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim

        # Compute query and key-value projection sizes
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Scaling factor for query
        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        # Define linear projections for QKV and output
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj

        self.attn_type = attn_type
        self.sliding_window_size = sliding_window_size
        self.attn_logit_softcapping = attn_logit_softcapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for attention mechanism.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            freqs_cis (torch.Tensor): Precomputed rotary embeddings.
            kv_write_indices (torch.Tensor): Indices to update key-value cache.
            kv_cache (Tuple[torch.Tensor, torch.Tensor]): Cached keys and values.
            mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Attention output of shape (batch_size, seq_len, hidden_size).
        """
        batch_size, input_len, _ = hidden_states.shape

        # Project hidden states to QKV
        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape QKV for multi-head attention
        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings to Q and K
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        # Update key-value cache
        k_cache, v_cache = kv_cache
        # index_copy_(): self and source expected to have the same dtype, but got (self) Float and (source) BFloat16

        # Ensure k_cache and v_cache have the same dtype as xk and xv
        k_cache = k_cache.to(xk.dtype)
        v_cache = v_cache.to(xv.dtype)

        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        # Repeat keys and values for query heads if needed
        if self.num_kv_heads != self.num_heads:
            key = torch.repeat_interleave(k_cache, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(v_cache, self.num_queries_per_kv, dim=2)
        else:
            key = k_cache
            value = v_cache

        # Compute scaled dot-product attention
        q = xq.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        q.mul_(self.scaling)  # Scale query
        scores = torch.matmul(q, k.transpose(-1, -2))  # Attention scores

        # Apply sliding window masking if needed
        if self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING and self.sliding_window_size is not None:
            sliding_mask = torch.triu(torch.ones_like(mask), -self.sliding_window_size + 1)
            sliding_mask = sliding_mask * torch.tril(sliding_mask, self.sliding_window_size - 1)
            mask = torch.where(sliding_mask == 1, mask, -2.3819763e38)

        # Apply softcapping to scores
        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores) * self.attn_logit_softcapping

        # Add mask and compute attention probabilities
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # Compute attention output
        output = torch.matmul(scores, v)  # Attention applied to values
        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)

        # Project back to hidden size
        output = self.o_proj(output)
        return output