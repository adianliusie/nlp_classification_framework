import torch
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz

from src.trainers.system_loader import SystemLoader
from src.utils.dir_helper import DirHelper

class InterpretableLoader(SystemLoader):
    def __init__(self, exp_path:str):
        self.exp_path = exp_path
        self.dir = DirHelper.load_dir(exp_path)
        super().set_up_helpers()

    def integrad(self, data_name:str, mode:str='test', idx:int=0):
        #load example to run interpretability on
        data_set = self.data_loader.prep_split(data_name=data_name, mode=mode)
        data_set = list(self.batcher(data_set, bsz=1))
        ex = data_set[idx]
        
        #create input and baseline tensors
        baseline_ids = self.create_baseline(ex)
        baseline_ids = torch.LongTensor([baseline_ids]).to(self.device)
        input_ids    = torch.LongTensor([ex.input_ids]).to(self.device)
        
        pred_scores = self.model(input_ids=input_ids).logits[0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        
        #set up the ingtegrated gradients
        lig = LayerIntegratedGradients(self.forward_interpret, self.model.transformer.embeddings.word_embeddings)

        attributions, delta = lig.attribute(inputs=input_ids,
                                            baselines=baseline_ids,
                                            return_convergence_delta=True, 
                                            internal_batch_size=8)
        
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)

        #visualise the saliency attribution
        vis = viz.VisualizationDataRecord(
                attributions,
                torch.softmax(pred_scores, dim=0)[1],
                torch.argmax(pred_scores),
                str(ex.label),
                torch.argmax(pred_scores),
                attributions.sum(),       
                tokens,
                delta)
        
        viz.visualize_text([vis])                
        print(ex.label)
        
    def forward_interpret(self, input_ids:torch.LongTensor):
        output = self.model(input_ids=input_ids)
        y = output.logits.max(1).values
        return y
    
    def create_baseline(self, ex):
        pad_idx  = self.tokenizer.pad_token_id 
        format_tokens = [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]
        baseline_ids = [(tok if tok in format_tokens else pad_idx) for tok in ex.input_ids]
        return baseline_ids
    
    @property
    def tokenizer(self): 
        return self.data_loader.tokenizer
    