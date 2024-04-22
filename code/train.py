import torch.cuda as cuda
import torch.nn as nn
import torch
import json
import os
from data_process import data_sampler
from transformers import BertTokenizer
from prototype_prompt import Protonets
from utils import *


if __name__ == "__main__":
    
    with open("config/config_new.json") as f:
        config = json.loads(f.read())

    device = torch.device('cuda', config["device_idx"])
    config['device'] = device
    config['n_gpu'] = torch.cuda.device_count()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    donum = 1
    max_len = 128
    hidden_layer_dim = 768

    for m in range(donum):

        sampler = data_sampler(config, tokenizer)
        sequence_results = []
        result_whole_test = []

        for i in range(6):
            
            num_class = len(sampler.id2rel)
            set_seed(config, config['random_seed'] + 10 * i)
            sampler.set_seed(config['random_seed'] + 10 * i)
            saveseen_relations = []
            proto_memory = {}
            prompt_memory = {}
            test_data = []
            p2_map = {}
            

            for steps, (training_data, valid_data, test_data, test_all_data, seen_relations, current_relations) in enumerate(sampler):

                savetest_all_data = test_all_data
                saveseen_relations = seen_relations
                currentnumber = len(current_relations)
                p2_map[steps] = []
                for relation in current_relations:
                    p2_map[steps].append(relation)
                
                divide_train_set = {relation: [] for relation in current_relations}
                for data in training_data:
                    divide_train_set[data[0]].append(data)

                divide_test_set = {relation: [] for relation in seen_relations}
                for data in test_all_data:
                    if data[0] in seen_relations:
                        divide_test_set[data[0]].append(data)

                p2_map[steps] = current_relations
                

                
                
                if steps == 0:
                    # base session
                    print("------- Training base stage started -------")
                    protonets = Protonets(Nc = 10,
                                    log_data = 'pretrained-PLM.pkl',
                                    config = config,
                                    step = steps,
                                    prompt_pool = prompt_memory,
                                    proto_pool = proto_memory,
                                    p2_map = p2_map)
                    protonets.train_base(divide_train_set,divide_test_set,current_relations,seen_relations)
                                
                else:
                    # cintinual session
                    print(f"------- {str(steps)} stage started -------")
                    s = steps-1
                    prompt_memory,proto_memory = load_center(f'center/prompt_dict{s}',f'center/proto_dict{s}')
                    protonets = Protonets(Nc = 10,
                                    log_data = f'./model/model_0.pkl',
                                    config = config,
                                    step = steps,
                                    prompt_pool = prompt_memory,
                                    proto_pool = proto_memory,
                                    p2_map = p2_map)
                    protonets.train_continue(divide_train_set,divide_test_set,current_relations,seen_relations)
                    
