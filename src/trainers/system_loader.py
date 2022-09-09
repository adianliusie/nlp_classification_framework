import torch.nn.functional as F
import numpy as np
import os

from tqdm import tqdm
from typing import List

from .QA_trainer import Trainer
from ..utils.torch_utils import no_grad
from ..utils.dir_helper import DirHelper
from ..data_utils.data_loader import QaDataLoader

class SystemLoader(Trainer):
    """Base loader class- the inherited class inherits
       the Trainer so has all experiment methods"""

    def __init__(self, exp_path:str):
        self.dir = DirHelper.load_dir(exp_path)
        
    def set_up_helpers(self):
        #load training arguments and set up helpers
        args = self.dir.load_args('model_args.json')
        super().set_up_helpers(args)

        #load final model
        self.load_model()
        self.model.eval()
        self.device = 'cuda:0'
        self.to(self.device)

    def load_preds(self, data_name:str, mode)->dict:
        probs = self.load_probs(data_name, mode)
        preds = {}
        for ex_id, probs in probs.items():
            preds[ex_id] = int(np.argmax(probs, axis=-1))  
        return preds
        
    def load_probs(self, data_name:str, mode)->dict:
        """loads predictions if saved, else generates"""
        if not self.dir.probs_exists(data_name, mode):
            self.set_up_helpers()
            self.generate_probs(data_name, mode)
        probs = self.dir.load_probs(data_name, mode)
        return probs

    def generate_probs(self, data_name:str, mode):
        probabilties = self._probs(data_name, mode)
        self.dir.save_probs(probabilties, data_name, mode)

    @no_grad
    def _probs(self, data_name:str, mode='test'):
        """get model predictions for given data"""
        self.model.eval()
        self.to(self.device)
        eval_data = self.data_loader.prep_MCRC_split(data_name, mode)
        eval_batches = self.batcher(data=eval_data, bsz=1, shuffle=False)
        
        probabilties = {}
        for batch in tqdm(eval_batches):
            ex_id = batch.ex_id[0]
            output = self.model_output(batch)

            logits = output.logits.squeeze(0)
            if logits.shape and logits.shape[-1] > 1:  # Get probabilities of predictions
                prob = F.softmax(logits, dim=-1)
            probabilties[ex_id] = prob.cpu().numpy()
        return probabilties
    
    def load_labels(self, data_name:str, mode='test', lim=None)->dict:
        eval_data = QaDataLoader.load_split(data_name, mode)
        labels_dict = {}
        for ex in eval_data:
            labels_dict[ex.ex_id] = ex.answer
        return labels_dict

    def load_inputs(self, data_name:str, mode='test')->dict:
        eval_data = QaDataLoader.load_split(data_name, mode)
        inputs_dict = {}
        for ex in eval_data:
            inputs_dict[ex.ex_id] = ex
        return inputs_dict
    
    def get_eval_data(self, data_name:str, mode='test'):
        return self.data_loader.prep_MCRC_split(data_name, mode)

class EnsembleLoader(SystemLoader):
    def __init__(self, exp_path:str):
        self.exp_path = exp_path
        self.paths  = [f'{exp_path}/{seed}' for seed in os.listdir(exp_path) if os.path.isdir(f'{exp_path}/{seed}')]
        self.seeds  = [SystemLoader(seed_path) for seed_path in self.paths]
    
    def load_probs(self, data_name:str, mode)->dict:
        seed_probs = [seed.load_probs(data_name, mode) for seed in self.seeds]

        ex_ids = seed_probs[0].keys()
        assert all([i.keys() == conv_ids for i in seed_probs])

        ensemble = {}
        for ex_id in ex_ids:
            probs = [seed[ex_id] for seed in seed_probs]
            probs = np.mean(probs, axis=0)
            ensemble[ex_id] = probs
        return ensemble    
    
    def load_preds(self, data_name:str, mode)->dict:
        probs = self.load_probs(data_name, mode)
        preds = {}
        for ex_id, probs in probs.items():
            preds[ex_id] = int(np.argmax(probs, axis=-1))  
        return preds
        
