import random
import numpy as np
import wordninja


class data_sampler(object):
    
    def __init__(self, config=None, tokenizer=None, max_length=128, blank_padding=False):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.blank_padding = blank_padding
        
        ##read train valid test ori_data
        self.training_data = self._gen_data(config['training_file'])
        self.valid_data = self._gen_data(config['valid_file'])
        self.test_data = self._gen_data(config['test_file'])

        ##load relation
        self.relation_names, self.id2rel = self._read_relations(config['relation_file'])
        self.num_clusters = config['num_clusters']
        
        self.cluster_labels = {}
        rel_index = np.load("dataset/fewrel/rel_index.npy")
        rel_cluster_label = np.load(config["rel_cluster_label"])
        for index, i in enumerate(rel_index):
            self.cluster_labels[i] = rel_cluster_label[index]
        self.splited_training_data = self._split_data(self.training_data, self.cluster_labels, self.num_clusters)
        self.splited_valid_data = self._split_data(self.valid_data, self.cluster_labels, self.num_clusters)
        self.splited_test_data = self._split_data(self.test_data, self.cluster_labels, self.num_clusters)
        self.seed = None

    def _split_data(self, data_set, cluster_labels, num_clusters):
        splited_data = [[] for _ in range(num_clusters)]
        for data in data_set:
            splited_data[cluster_labels[int(data[0])]].append(data)
        return splited_data
        
    def set_seed(self, seed):
        self.seed = seed
        
    # 该函数作为最基础的根据数据文件加载列表数据
    def _gen_data(self, file):
        sample_data = []
        with open(file) as file_in:
            for line in file_in:
                items = line.strip().split('\t')
                relation_ix = items[0]
                text = items[1]
                head = items[2]
                tail = items[3]
                template = f'{text} [SEP] {head} [MASK] {tail}'
                tem = self.tokenizer.tokenize(template)
                index = tem.index('[MASK]')+1
                sample_data.append([relation_ix, text, head, tail, template, index])
        return sample_data
    
    # 以下三个函数功能基本完全一致，都是为了读取关系文件中的关系
    def _read_relations(self, file):
        relation_list = [self._split_relation_into_words(self._remove_return_sym('fill fill fill'))]
        id2rel = {0: 'fill fill fill'}
        with open(file) as file_in:
            for line in file_in:
                relation_list.append(self._split_relation_into_words(self._remove_return_sym(line)))
                id2rel[len(id2rel)] = self._remove_return_sym(line)
        return relation_list, id2rel
    
    def _split_relation_into_words(self, relation):
        word_list = []
        for word_seq in relation.split("/")[-3:]:
            for word in word_seq.split("_"):
                word_list += wordninja.split(word)
        return " ".join(word_list)
    
    def _remove_return_sym(self, str):
        return str.split('\n')[0]
    
    def __iter__(self):
        return sequence_data_sampler(self, self.seed)
