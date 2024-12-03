import torch
import torch.nn as nn

class RMSNorm(torch.nn.Module):
    """
    Implements Root Mean Square Layer Normalization (RMSNorm).
    
    RMSNorm normalizes the input tensor based on its root mean square (RMS) values.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        """
        Initializes the RMSNorm layer.

        Args:
            dim (int): Dimension of the input tensor to normalize.
            eps (float): A small value added for numerical stability (default: 1e-6).
            add_unit_offset (bool): Whether to add 1 to the learnable weight before scaling.
        """
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        # Learnable parameter for scaling the normalized values
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        """
        Normalizes the input tensor based on its root mean square (RMS) values.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: RMS-normalized tensor.
        """
        # Compute RMS norm: x / sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Applies RMS normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Normalized and scaled tensor.
        """
        # Normalize the input tensor to float32 for stability
        output = self._norm(x.float())
        
        # Scale the normalized tensor using the learnable weight
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        
        # Cast the output back to the input tensor's data type
        return output.type_as(x)
    
    def load_from_path(self, path):
        # Yolu belirtilen tensörü yükle
        weight_tensor = torch.load(path)
        # Doğru cihazda olduğundan emin ol
        weight_tensor = weight_tensor.to(self.weight.device)
        # Tensörü nn.Parameter olarak sar
        self.weight = nn.Parameter(weight_tensor)