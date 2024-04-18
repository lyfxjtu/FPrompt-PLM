from torch.autograd import Variable
import numpy as np
import random
import torch.nn as nn
import pickle
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from transformers import BertTokenizer, BertModel
from prompt import SoftEmbedding
import torch.autograd as autograd
import datetime
from utils import *
from sklearn.manifold import TSNE

def eucli_tensor(x,y):
    return -1*torch.sqrt(torch.sum((x-y)*(x-y))).view(1)

def prompt_initial(model):
    prompt = model.state_dict()["embeddings.word_embeddings.learned_embedding"].detach().clone()
    prompt = nn.init.xavier_uniform_(prompt)
    model_state_dict = model.state_dict()
    model_state_dict["embeddings.word_embeddings.learned_embedding"].copy_(prompt)
    return model_state_dict

def prompt_similarity(x,y):
    x = torch.tensor(x)
    y = torch.tensor(y)
    return -1 * torch.sqrt(torch.sum((x - y) * (x - y))).view(1)

class ProtoRepre(nn.Module):
    
    def __init__(self, hidden_state, multi_head, device):
        super(ProtoRepre,self).__init__()
        self.hidden_state = hidden_state
        self.atten = nn.MultiheadAttention(hidden_state, multi_head).to(device)
        
    def forward(self, data_embedding_i):

        avg_proto_i = torch.sum(data_embedding_i, 0) / data_embedding_i.shape[0]
        attn_input = torch.cat((avg_proto_i.unsqueeze(0), data_embedding_i), dim=0).unsqueeze(1)
        attn_output, attn_output_weights = self.atten(attn_input, attn_input, attn_input)
        proto_embedding_i = attn_output[0] + avg_proto_i.unsqueeze(0)
        return proto_embedding_i, attn_output_weights

class Protonets(object):
    
    def __init__(self,Nc,log_data,config,step,prompt_pool,proto_pool,p2_map):
        
        # Model parameter initialization and loading.
        self.step = step
        self.Ns = config['Ns']
        self.Nq = config['Nq']
        self.Nc = Nc
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Define a dictionary for the prototype centers.
        self.proto_pool = proto_pool
        self.prompt_pool = prompt_pool
        self.p2map = p2_map
        self.hidden_state = 768
        self.proto = ProtoRepre(self.hidden_state, self.config['multi_head'], self.config["device"])
        
        # Specify adjustable parameters for the "continue" stage.
        self.unfreeze_layers = self.config["unfreeze_layers_continue"]
        
        # Inherit the model directly from the base stage.
        self.model = torch.load(log_data , map_location=self.config["device"])
        self.test_model = torch.load(log_data , map_location=self.config["device"])
        if step == 1:
            with torch.no_grad():
                self.prompt_pool[0] = self.model.state_dict()["embeddings.word_embeddings.learned_embedding"].detach().clone()

        # Randomly initialize the soft prompt parameters.
        model_state_dict = prompt_initial(self.model)
        self.model.load_state_dict(model_state_dict)
        
        # After loading the model, verify the adjustable parameters.
        for name ,param in self.model.named_parameters():
            param.requires_grad = any(ele in name for ele in self.unfreeze_layers)

    def randomSample(self,D_set):
        # Randomly extract a support set and query set from the given data: (where the number of supports is fixed to self.Ns, and the rest are query samples).
        index_list = list(range(len(D_set)))
        random.shuffle(index_list)
        support_data_index = index_list[:self.Ns]
        query_data_index = index_list[-self.Nq:]
        support_set = [D_set[i] for i in support_data_index]
        query_set = [D_set[i] for i in query_data_index]
        return support_set,query_set

    def encoder(self,data_set):
        # Define the encoding process (embedding) of the data.
        result = []
        for i in range(len(data_set)):
            data = data_set[i]
            encode_input = self.tokenizer(data[4], 
                                        return_tensors='pt',
                                        padding = True, 
                                        max_length = 128, 
                                        truncation = True)
            encode_input['input_ids'] = torch.cat([torch.full((1,self.config["n_tokens"]), 0), encode_input['input_ids']], 1)
            encode_input['attention_mask'] = torch.cat([torch.full((1,self.config["n_tokens"]), 1), encode_input['attention_mask']], 1)
            encode_input['token_type_ids'] = torch.cat([torch.full((1,self.config["n_tokens"]), 0), encode_input['token_type_ids']], 1)
            encode_input = encode_input.to(self.config["device"])
            index = data[5]
            featrue = self.model(**encode_input)
            result.append(featrue[0][0][index + self.config['n_tokens']])
        return result
    
    def predict(self, data_set):
        data = []
        index = []
        for n in range(len(data_set)):
            data.append(data_set[n][4])
            index.append(data_set[n][5])

        encode_input = self.tokenizer(data, return_tensors='pt',padding = True, max_length = 128, truncation = True)
        encode_input['input_ids'] = torch.cat([torch.full((100,self.config["n_tokens"]), 0), encode_input['input_ids']], 1)
        encode_input['attention_mask'] = torch.cat([torch.full((100,self.config["n_tokens"]), 1), encode_input['attention_mask']], 1)
        encode_input['token_type_ids'] = torch.cat([torch.full((100,self.config["n_tokens"]), 0), encode_input['token_type_ids']], 1)
        encode_input = encode_input.to(self.config["device"])
        
        featrue_q = [[0 for _ in range(len(self.prompt_pool))] for _ in range(100)]
        for k in range(len(self.prompt_pool)):
            with torch.no_grad():
                layer_params_dict = self.test_model.state_dict()
                layer_params_dict["embeddings.word_embeddings.learned_embedding"].copy_(self.prompt_pool[k])
                self.test_model.load_state_dict(layer_params_dict)
                featrue = self.test_model(**encode_input).last_hidden_state.detach().clone()

            for t in range(len(index)):
                featrue_q[t][k] = featrue[t][index[t] + self.config['n_tokens']]
        
        return featrue_q
    
    def compute_center(self,data_set):
        ini_embedding = self.encoder(data_set)
        proto_input = torch.stack(list(ini_embedding))
        return torch.mean(proto_input, dim=0)
    
    def compute_center_transformer(self,data_set):
        # Compute the prototype centers by stacking the encoded results from a list into a tensor, which serves as the input to the transformer.
        ini_embedding = self.encoder(data_set,mode=0)
        proto_input = torch.stack(list(ini_embedding))
        return self.proto(proto_input)
    
    def momentum_center(self,label,data_set):
        
        para = self.config["momentum_params"]
        ini_embedding = self.encoder(data_set)
        proto_input = torch.stack(list(ini_embedding))
        proto_output = torch.mean(proto_input, dim=0)
        proto_momentum = self.proto_pool[label].detach()
        return (1-para)*proto_momentum + para*proto_output
    
    def train_continue(self,train_data,test_data,current_class_list,seen_relation):
        
        best_test = 0
        loss_weight = self.config["loss_weight_continue"]
        choss_class_index = current_class_list
        query_set = []
        optimizer_encoder = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        # optimizer_proto = torch.optim.Adam(filter(lambda p: p.requires_grad, self.proto.parameters()), lr=0.001)
        
        for n in range(self.config["continue_epoch"]):
            self.model.train()
            query_set = []
            for label in choss_class_index:
                D_set = train_data[label]
                support,query = self.randomSample(D_set)
                self.proto_pool[label] = self.compute_center(support)
                # self.proto_pool[label] = (
                #     self.compute_center(support)
                #     if self.proto_pool.get(label) is None
                #     else self.momentum_center(label, support)
                # )
                query_set.append(query)


            optimizer_encoder.zero_grad()
            # optimizer_proto.zero_grad()

            # Define the loss function.
            loss_1 = self.loss_cls_continue(query_set,current_class_list)
            loss_2 = self.loss_self(current_class_list)

            loss_continue = loss_weight[0]*loss_1 + loss_weight[1]*loss_2
            print(f"loss_continue:{loss_continue}loss_cls_continue:{loss_1}loss_prompt:{loss_2}")
            loss_continue.backward()
            
            # grad_loss_base = autograd.grad(loss_base, self.model.parameters(), retain_graph=True)
            # for param, grad in zip(self.encoder.parameters(), grad_loss_base):
            #     param.grad = grad.detach()
            # Modify the corresponding parameters.
            optimizer_encoder.step()
            
            if n % 5 == 0:
                self.model.eval()
                train_acc = self.test_now(train_data,current_class_list)
                test_acc = self.test_now(test_data,current_class_list)
                print(train_acc,test_acc)
                test_accury = self.test(test_data,seen_relation)
                with torch.no_grad():
                    self.prompt_pool[self.step] = self.model.state_dict()["embeddings.word_embeddings.learned_embedding"].detach().clone()
                if best_test < test_accury:
                    best_test = test_accury
                    self.save_center(f'center/{str(datetime.date.today())}/prompt_dict{self.step}',f'center/proto_dict{self.step}')
                str_data = f'{str(datetime.datetime.now())}:The accuracy of step {str(self.step)}eposide {n}is {test_accury},Best accuracy is {best_test}' + '\n'
                with open(f'./log/2_model_step_eval_{str(datetime.date.today())}.txt', "a") as f:
                    f.write(str_data)
                print(f'The accuracy of step {str(self.step)}eposide {n}is {test_accury},Best accuracy is {best_test}')

    def loss_cls_continue(self,query_set,current_relation):
        # Concatenate the query set and class prototypes into tensors separately.
        loss_cls = autograd.Variable(torch.FloatTensor([0])).to(self.config["device"])

        proto_pool = []
        proto_pool.extend(self.proto_pool[relation] for relation in current_relation)
        for i in range(self.Nc):
            query = query_set[i]
            query_embedding = self.encoder(query)
            for j in range(self.Nq):
                predict = torch.zeros(self.Nc)
                for k, proto in enumerate(proto_pool):
                    predict[k] = eucli_tensor(query_embedding[j],proto)
                loss_cls = loss_cls - F.log_softmax(predict,dim=0)[i]
        loss_cls /= self.Nq*self.Nc
        return loss_cls
    
    def loss_cls_all(self,query_set,seen_relation):
        loss_cls = autograd.Variable(torch.FloatTensor([0])).to(self.config["device"])

        proto_pool = []
        proto_pool.extend(self.proto_pool[relation] for relation in seen_relation)
        for i in range(self.Nc):
            query = query_set[i]
            query_embedding = self.encoder(query)
            for j in range(self.Nq):
                predict = torch.zeros(self.Nc)
                for k, proto in enumerate(proto_pool):
                    predict[k] = eucli_tensor(query_embedding[j],proto)
                loss_cls = loss_cls - F.log_softmax(predict,dim=0)[i]
        loss_cls /= self.Nq*self.Nc
        return loss_cls
    
    def loss_prompt_weight(self,prompt_pool,proto_pool):
        loss_prompt = autograd.Variable(torch.FloatTensor([0])).to(self.config["device"])
        for i in range(self.step-1):
            for j in range(self.Nc):
                for k in range(self.Nc):
                    loss_prompt += prompt_similarity(prompt_pool[self.step],prompt_pool[i])*eucli_tensor(proto_pool[self.p2map[self.step][j]],proto_pool[self.p2map[i][k]])
        return loss_prompt
    
    def loss_cross(self,current_relation,seen_relation,proto_pool):
        loss_cross = autograd.Variable(torch.FloatTensor([0])).to(self.config["device"])
        
        return loss_cross
    
    def loss_self(self,current_relation):
        loss_self = autograd.Variable(torch.FloatTensor([0])).to(self.config["device"])
        # loss_self = torch.tensor(0.0, device=self.config["device"], requires_grad=True)
        for i in range(self.Nc):
            for j in range(i+1,self.Nc):
                dist = eucli_tensor(self.proto_pool[current_relation[i]],self.proto_pool[current_relation[j]])
                sim = torch.exp(dist)
                loss_self = loss_self + sim.item()
        return loss_self
            

    def test(self,test_data,seen_relations):
        test_accury = []
        proto_pool = []
        proto_pool.extend(self.proto_pool[relation] for relation in seen_relations)
        query_set = []
        for label in seen_relations:
            D_set = test_data[label]
            index_list = list(range(len(D_set)))
            query = []
            query.extend(D_set[i] for i in index_list)
            query_set.append(query)
        
        for i in range(len(query_set)):
            featrue_q = self.predict(query_set[i])
            
            for o in range(100):
                for j in range(0,10*len(self.prompt_pool)):
                    k = j//10
                    center_j = proto_pool[j]
                    if j == 0:
                        predict = eucli_tensor(featrue_q[o][k],center_j)
                    else:
                        predict = torch.cat((predict, eucli_tensor(featrue_q[o][k],center_j)), 0)
                y_pre_j = int(torch.argmax(F.log_softmax(predict,dim=0)))
                test_accury.append(1 if y_pre_j == i else 0)
                
        return sum(test_accury)/len(test_accury)
    
    def test_now(self,teat_data,current_relation):
        test_acc = []
        proto_pool = []
        proto_pool.extend(self.proto_pool[relation] for relation in current_relation)
        query_set = []
        for label in current_relation:
            query = teat_data[label]
            query_set.append(query)
            
        for i in range(self.Nc):
            query = query_set[i]
            query_embedding = self.encoder(query)
            for j in range(len(query)):
                predict = torch.zeros(self.Nc)
                for k, proto in enumerate(proto_pool):
                    predict[k] = eucli_tensor(query_embedding[j],proto)
                y_pre_j = int(torch.argmax(F.log_softmax(predict,dim=0)))
                test_acc.append(1 if y_pre_j == i else 0)
        # print(sum(test_acc),len(test_acc))
        
        return sum(test_acc)/len(test_acc)
        
    def save_center(self,path1,path2):
        # prompt = self.prompt_pool
        # proto = self.proto_pool
        proto = {}
        prompt = {}
        for label in self.proto_pool:
            proto[label] = self.proto_pool[label].detach()
        for label in self.prompt_pool:
            prompt[label] = self.prompt_pool[label].detach()
        with open(path1,'wb') as f:
            pickle.dump(prompt,f)
        with open(path2,'wb') as f:
            pickle.dump(proto,f)
