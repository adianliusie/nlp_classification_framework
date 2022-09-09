import os
import json
import pickle
import sys

#== File handling functions ================================================
def save_json(data:dict, path:str):
    with open(path, "x") as outfile:
        json.dump(data, outfile, indent=2)

def load_json(path:str)->dict:
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    return data

def save_pickle(data, path:str):
    with open(path, 'xb') as output:
        pickle.dump(data, output)

def load_pickle(path:str):
    with open(path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

#== Location utils ==========================================================
def get_base_dir():
    """automatically gets root dir of framework"""
    #gets path of the src folder 
    cur_path = os.path.abspath(__file__)
    src_path = cur_path.split('/src')[0] + '/src'
    
    #can be called through a symbolic link, if so go out one more dir.
    if os.path.islink(src_path):
        src_path = os.path.abspath(os.readlink(src_path))
    
    base_path = src_path.split('/src')[0]    
    return base_path

#== Logging utils ===========================================================
def save_script_args():
    CMD = f"python {' '.join(sys.argv)}\n"
    with open('CMDs', 'a+') as f:
        f.write(CMD)
       
