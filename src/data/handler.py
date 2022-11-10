import random

from typing import List
from types import SimpleNamespace
from tqdm import tqdm
from copy import deepcopy
from functools import lru_cache

from models.tokenizers import load_tokenizer
from .load_classification_cad import load_cad_cls_data, CAD_CLS_DATA
from .load_classification_hf import load_hf_cls_data, HF_CLS_DATA
from .load_nli_hf import load_hf_nli_data, HF_NLI_DATA

#== Main DataHandler class ========================================================================#
class DataHandler:
    def __init__(self, trans_name:str):
        self.tokenizer = load_tokenizer(trans_name)
    
    #== Data processing (i.e. tokenizing text) ====================================================#
    @lru_cache(maxsize=5)
    def prep_split(self, data_name:str, mode:str, lim=None):
        data = self.load_split(data_name, mode, lim)
        if is_classification(data_name): output = self._prep_ids(data) 
        elif is_nli(data_name): output = self._prep_ids_pairs(data) 
        else: raise ValueError(f"invalid data set: {data_name}")
        return output
    
    @lru_cache(maxsize=5)
    def prep_data(self, data_name, lim=None):
        train, dev, test = self.load_data(data_name=data_name, lim=lim)
        if is_classification(data_name):
            train, dev, test = [self._prep_ids(split) for split in [train, dev, test]]
        elif is_nli(data_name):
            train, dev, test = [self._prep_ids_pairs(split) for split in [train, dev, test]]
        else: raise ValueError(f"invalid data set: {data_name}")
        return train, dev, test
        
    def _prep_ids(self, split_data:List[SimpleNamespace]):
        split_data = deepcopy(split_data)
        for ex in tqdm(split_data):
            ex.input_ids = self.tokenizer(ex.text).input_ids
        return split_data

    def _prep_ids_pairs(self, split_data):
        split_data = deepcopy(split_data)
        for ex in tqdm(split_data):
            text_1_ids = self.tokenizer(ex.text_1).input_ids
            text_2_ids = self.tokenizer(ex.text_2).input_ids
            ex.input_ids = text_1_ids + text_2_ids[1:]
        return split_data
    
    #== Data loading utils ========================================================================#
    @classmethod
    def load_split(cls, data_name:str, mode:str, lim=None):
        split_index = {'train':0, 'dev':1, 'test':2}
        data = cls.load_data(data_name, lim)[split_index[mode]]
        return data
    
    @staticmethod
    @lru_cache(maxsize=5)
    def load_data(data_name:str, lim=None):
        if   data_name in HF_CLS_DATA : train, dev, test = load_hf_cls_data(data_name)
        elif data_name in CAD_CLS_DATA: train, dev, test = load_cad_cls_data(data_name)
        elif data_name in HF_NLI_DATA : train, dev, test = load_hf_nli_data(data_name)
        else: raise ValueError(f"invalid dataset name: {data_name}")
            
        if lim:
            train = rand_select(train, lim)
            dev   = rand_select(dev, lim)
            test  = rand_select(test, lim)
            
        train, dev, test = to_namespace(train, dev, test)
        return train, dev, test
    
#== Misc utils functions ============================================================================#
def is_classification(data_name:str):
    return data_name in HF_CLS_DATA + CAD_CLS_DATA

def is_nli(data_name:str):
    return data_name in HF_NLI_DATA

def rand_select(data:list, lim:None):
    if data is None: return None
    random_seed = random.Random(1)
    data = data.copy()
    random_seed.shuffle(data)
    return data[:lim]

def to_namespace(*args:List):
    def _to_namespace(data:List[dict])->List[SimpleNamespace]:
        return [SimpleNamespace(ex_id=k, **ex) for k, ex in enumerate(data)]

    output = [_to_namespace(split) for split in args]
    return output if len(args)>1 else output[0]

def get_num_classes(data_name:str):
    if is_classification(data_name): output = 2
    elif is_nli(data_name)         : output = 3
    return output 
    