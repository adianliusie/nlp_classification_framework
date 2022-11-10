from typing import List, Dict, Tuple, TypedDict

class SingleText(TypedDict):
    """Output example formatting (only here for documentation)"""
    text : str
    label : int

CAD_CLS_DATA = ['imdb', 'rt', 'sst', 'yelp', 'boolq']

def load_cad_cls_data(data_name)->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    """ loading sentiment classification datsets used in Kaushik et. Al 2021,
        'Learning the Difference that Makes a Difference with Counterfactually Augmented Data' """
    if   data_name == 'imdb-small':  train, dev, test = load_imdb()
    else: raise ValueError(f"invalid single text dataset name: {data_name}")
    return train, dev, test
    