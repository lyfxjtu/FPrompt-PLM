import random
import torch
import numpy as np
import pickle

def set_seed(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config['n_gpu'] > 0 and torch.cuda.is_available() and config['use_gpu']:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def load_center(path1,path2):
    
    with open(path1,'rb') as f:
        prompt_momery = pickle.load(f)
    with open(path2,'rb') as f:
        proto_momery = pickle.load(f)
    
    return prompt_momery,proto_momery

def load_center0(path1,path2):
    
    with open(path1,'rb') as f:
        prompt_momery = pickle.load(f)
    with open(path2,'rb') as f:
        proto_momery = pickle.load(f)
    
    return prompt_momery,proto_momery
        
