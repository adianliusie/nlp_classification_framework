import torch
import torch.nn as nn
from types import SimpleNamespace

from ..utils.torch_utils import load_transformer

class TransformerModel(torch.nn.Module):
    """basic transformer model for multi-class classification""" 
    def __init__(self, trans_name:str, num_classes:int=2):
        super().__init__()
        self.transformer = load_transformer(trans_name)
        h_size = self.transformer.config.hidden_size
        self.output_head = nn.Linear(h_size, num_classes)
        
    def forward(self, *args, **kwargs):
        trans_output = self.transformer(*args, **kwargs)
        H = trans_output.last_hidden_state  #[bsz, L, 768] 
        h = H[:, 0]                         #[bsz, 768] 
        logits = self.output_head(h)             #[bsz, C] 
        return SimpleNamespace(h=h, logits=logits)
    
class TransformerHydraModel(torch.nn.Module):
    """investigation of using multi-head model"""
    def __init__(self, trans_name:str, num_classes:int=2):
        self.transformer = load_transformer(trans_name)
        h_size = self.transformer.config.hidden_size
        self.output_heads = [nn.Linear(h_size, num_classes) for _ in range(12)]
    
    def forward(self, *args, **kwargs):
        trans_output = self.transformer(*args, **kwargs)
        layer_outpus = trans_output.hidden_states
        H = [layer[0, :] for layer in layer_outpus]
        logits_list = [head(h) for head, h in zip(self.output_heads, H)]
        return SimpleNamespace(h=H[0], logits=logits_list[0],
                               H=H, logits_list=logits_list)
