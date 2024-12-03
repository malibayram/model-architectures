import torch
import torch.nn as nn
from typing import Union, Optional, Tuple

class Sampler(nn.Module):
    """
    A class to sample the next token based on logits generated from model outputs.
    """

    def __init__(self, vocab_size: int):
        """
        Initializes the Sampler class.
        
        Args:
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.vocab_size = vocab_size

    @torch.no_grad()
    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the next token IDs based on model outputs.

        Args:
            embedding (torch.Tensor): The embedding matrix.
            hidden_states (torch.Tensor): Hidden states of the model.
            output_positions (torch.Tensor): The positions of the tokens to process.
            temperatures (Optional[torch.Tensor]): Controls sampling randomness.
            top_ps (torch.Tensor): Top-p values for nucleus sampling.
            top_ks (torch.Tensor): Top-k values to restrict to the most probable tokens.
            embedding_bias (Optional[torch.Tensor]): Bias added to logits.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The sampled next token IDs and logits.

        Functionality:
            1. Logit Calculation:
                - Multiplies hidden states with the embedding matrix to calculate logits.
                - Optionally adds bias to the logits if `embedding_bias` is provided.
            2. Temperature Scaling:
                - Divides logits by the temperature values to adjust sampling randomness.
            3. Probability Calculation:
                - Applies softmax to calculate probabilities from logits.
                - Sorts probabilities and applies top-p (nucleus) and top-k filtering.
            4. Next Token Sampling:
                - Samples next token IDs from filtered probabilities using `torch.multinomial`.
        """
        # Select hidden states at the specified output positions
        hidden_states = hidden_states[torch.arange(hidden_states.size(0)), output_positions]

        # Compute logits by multiplying hidden states with the embedding matrix
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias

        # If no temperature is provided, return the token with the highest logit
        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1), logits

        # Scale logits by temperature
        logits.div_(temperatures.unsqueeze(dim=1))

        # Compute probabilities with softmax
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Apply top-p (nucleus) filtering
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        # Apply top-k filtering
        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalize probabilities after filtering
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))

        # Sample the next token IDs from probabilities
        next_token_ids = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(dim=-1)

        # Return the sampled token IDs and logits
        return next_token_ids, logits
