import torch.nn as nn
import torch.nn.functional as F
from linear import Linear  # Custom Linear layer assumed to be defined in `linear.py`

class GemmaMLP(nn.Module):
    """
    Implements a Multi-Layer Perceptron (MLP) for Gemma with gating and projection layers.
    
    The MLP performs:
    1. Gating via `gate_proj`.
    2. Up projection via `up_proj`.
    3. Down projection via `down_proj`.
    """

    def __init__(
        self,
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
    ):
        """
        Initializes the GemmaMLP module.

        Args:
            hidden_size (int): Size of the last dimension of the input tensor.
            intermediate_size (int): Size of the intermediate projection.
        """
        super().__init__()
        # Linear layer for gating mechanism
        self.gate_proj = gate_proj
        # Linear layer for projecting the input up
        self.up_proj = up_proj
        # Linear layer for projecting the fused result back down
        self.down_proj = down_proj

    def forward(self, x):
        """
        Forward pass for the GemmaMLP.

        Args:
            x (torch.Tensor): Input tensor of shape (..., hidden_size).

        Returns:
            torch.Tensor: Output tensor of shape (..., hidden_size).
        """
        # Apply the gating projection and activation function
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")  # GELU activation for gating

        # Apply the up projection
        up = self.up_proj(x)

        # Fuse the gate and up projections element-wise
        fuse = gate * up

        # Project the fused result back to the original hidden size
        outputs = self.down_proj(fuse)

        return outputs