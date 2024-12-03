import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim)),
            requires_grad=False,
        )

    def forward(self, x):
        weight = self.weight
        output = F.embedding(x, weight)
        return output
    
    def load_from_path(self, path):        
        # Load the tensor from the specified path
        weight_tensor = torch.load(path)
        # Ensure it's on the correct device
        weight_tensor = weight_tensor.to(self.weight.device)
        # Wrap the tensor as a nn.Parameter
        self.weight = nn.Parameter(weight_tensor)