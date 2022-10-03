import torch
import torch.nn as nn
from types import SimpleNamespace

from ...utils.torch_utils import load_transformer
  
class TransformerHydraModel(torch.nn.Module):
    """investigation of using multi-head model"""
    def __init__(self, trans_name:str, num_classes:int=2):
        super().__init__()
        self.transformer = load_transformer(trans_name)
        h_size = self.transformer.config.hidden_size
        self.output_heads = torch.nn.ModuleList([nn.Linear(h_size, num_classes) for _ in range(12)])
    
    def forward(self, *args, **kwargs):
        trans_output = self.transformer(*args, output_hidden_states=True, **kwargs)
        layer_outputs = trans_output.hidden_states
        H = [layer[:,0] for layer in layer_outputs]
        logits_list = [head(h) for head, h in zip(self.output_heads, H)]
        return SimpleNamespace(h=H[0], logits=logits_list[0],
                               H=H, logits_list=logits_list)

    def freeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = False
      
    def unfreeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = True
            