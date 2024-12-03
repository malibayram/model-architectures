import torch  
from torch import nn  
from torch.nn import functional as F  

class Linear(nn.Module):  
    def __init__(self, in_features: int, out_features: int):  
        """  
        in_features: Giriş özelliklerinin boyutu  
        out_features: Çıkış özelliklerinin boyutu  
        """  
        super().__init__()  
        self.weight = nn.Parameter(  
            torch.empty((out_features, in_features)),  
            requires_grad=False,  # Gradyanlar hesaplanmayacak
        )  

    def forward(self, x):  
        """  
        x: Giriş tensörü  
        """  
        weight = self.weight  
        output = F.linear(x, weight)  # y = xW^T  
        return output  
    
    def load_from_path(self, path):
        # Yolu belirtilen tensörü yükle
        weight_tensor = torch.load(path)
        # Doğru cihazda olduğundan emin ol
        weight_tensor = weight_tensor.to(self.weight.device)
        # Tensörü nn.Parameter olarak sar
        self.weight = nn.Parameter(weight_tensor)