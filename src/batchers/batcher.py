# Adian Liusie, 2022-SEP-02

import torch
import random

from typing import List
from types import SimpleNamespace

class Batcher:
    def __init__(self, max_len:int=512, device:str='cuda'):
        self.max_len = max_len
        self.device  = device

    def batches(self, data:list, bsz:int, shuffle:bool=False):
        """splits the data into batches and returns them"""
        examples = self._prep_examples(data)
        if shuffle: random.shuffle(examples)
        batches = [examples[i:i+bsz] for i in range(0,len(examples), bsz)]
        for batch in batches:
            yield self.batchify(batch)
  
    def batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        ex_id, input_ids, labels = zip(*batch)  
        input_ids, attention_mask = self._get_padded_ids(input_ids)
        labels = torch.LongTensor(labels).to(self.device)
        return SimpleNamespace(ex_id=ex_id, 
                               input_ids=input_ids, 
                               attention_mask=attention_mask, 
                               labels=labels)
    
    def _prep_examples(self, data:list):
        """ sequence classification input data preparation"""
        prepped_examples = []
        for ex in data:
            ex_id = ex.ex_id
            label = ex.label
            input_ids = ex.input_ids
            
            # if ids larger than max size, then truncate
            if self.max_len and (len(input_ids)>self.max_len): 
                input_ids = input_ids[-self.max_len:]
            
            prepped_examples.append([ex_id, input_ids, label])
        return prepped_examples
                
    def _get_padded_ids(self, ids:list, pad_id:int=0)->(torch.LongTensor, torch.LongTensor):
        """ pads 2D input ids arry so that every row has the same length """
        max_len = max([len(x) for x in ids])
        padded_ids = [x     + [pad_id]*(max_len-len(x)) for x in ids]
        mask       = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask
        
    def to(self, device:torch.device):
        """ sets the device of the batcher """
        self.device = device
    
    def __call__(self, data, bsz, shuffle=False):
        """routes the main method do the batches function"""
        return self.batches(data=data, bsz=bsz, shuffle=shuffle)
    