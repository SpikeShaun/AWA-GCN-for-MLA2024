from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
from collections import Counter
import json
import pickle
import os
import torch
import re
import numpy as np
import collections

label2id = json.load(open('./data/label2id.json', encoding='utf-8'))
pospair2id = json.load(open('./pdata/pospair2id.json', encoding='utf-8'), )
deppair2id = json.load(open('./pdata/deppair2id.json', encoding='utf-8'), )
word2id = json.load(open('./pdata/word2id.json', encoding='utf-8'), )

class Tokenizer:
    def __init__(self, max_len=None, char_level=False):
        self.word2id = {'<pad>': 0, '<unk>': 1}
        self.id2word = {0: '<pad>', 1: '<unk>'}
        self.max_len = max_len
        self.char_level = char_level

    def token(self, texts):
        if isinstance(texts[0], list):
            return texts
        else:
            if self.char_level:
                return [list(text) for text in texts]
            else:
                return [text.split() for text in texts]

    def fit_on_texts(self, texts):
        texts = [x.lower() for x in texts]
        tokenized_texts = self.token(texts)
        # build word2id dictionary
        token_list = []
        for x in tokenized_texts:
            token_list += x
        word_counts = Counter(token_list)
        word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, count in word_counts:
            if word not in self.word2id:
                self.word2id[word] = len(self.word2id)
                self.id2word[self.word2id[word]] = word

    def texts_to_sequences(self, texts, max_len=None):
        if max_len is None:
            max_len = self.max_len
        texts = [x.lower() for x in texts]
        tokenized_texts = self.token(texts)
        sequences = []
        for tokens in tokenized_texts:
            sequence = [self.word2id.get(x, self.word2id['<unk>']) for x in tokens]
            sequences.append(sequence)
        padded_sequences = self.pad_sequences(sequences, max_len)
        return padded_sequences

    def pad_sequences(self, sequences, max_len=None):

        padded_sequences = []
        for sequence in sequences:
            if max_len is not None and len(sequence) > max_len:
                sequence = sequence[:max_len]
            pad_len = max_len - len(sequence) if max_len is not None else 0
            padded_sequence = sequence + [self.word2id['<pad>']] * pad_len
            padded_sequences.append(padded_sequence)
        return padded_sequences

    def save_vocab(self, word2id_file, id2word_file):
        with open(word2id_file, 'w', encoding='utf-8') as f:
            json.dump(self.word2id, f, ensure_ascii=False, indent=1)
        with open(id2word_file, 'w', encoding='utf-8') as f:
            json.dump(self.id2word, f, ensure_ascii=False, indent=1)
        print('word2id and id2word saved success............')

    def load_vocab(self, word2id_file, id2word_file):
        with open(word2id_file, 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
        with open(id2word_file, 'r', encoding='utf-8') as f:
            self.id2word = json.load(f)
        print('word2id and id2word loaded success............')


def read_data(file, ):
    data = json.load(open(file, encoding='utf-8'))
    return data

class PLMDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)




class CollatorGTS:
    def __init__(self, tokenizer, max_seq_length, use_aspect_class=False):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_aspect_class = use_aspect_class

    def __call__(self, batch):

        texts = [list(x['Sentence']) for x in batch]
        Aspect = [x['Aspect'] for x in batch]
        AspectFromTo = [x['AspectFromTo'] for x in batch]
        Opinion = [x['Opinion'] for x in batch]
        OpinionFromTo = [x['OpinionFromTo'] for x in batch]
        Category = [x['Category'] for x in batch]

        Intensity = [x['Intensity'] for x in batch]


        # 使用BERT tokenizer对所有文本进行编码
        inputs = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_seq_length,
                                return_tensors='pt', is_split_into_words=True)
        inputs.pop('token_type_ids')
        length = inputs['input_ids'].shape[1]

        triple_labels = torch.zeros(len(texts), length, length, dtype=torch.int64)

        score_labels = torch.zeros(len(texts), length, length, dtype=torch.float32)

        for k in range(len(batch)):
            for a, at, o, ot, c, intensity in list(zip(Aspect[k],
                                            AspectFromTo[k],
                                            Opinion[k],
                                            OpinionFromTo[k],
                                            Category[k],
                                            Intensity[k])):
                ats, ate = at.split('#')
                start_a, end_a = int(ats) - 1, int(ate) # 这个因为元数据索引是从0开始
                start_a, end_a = start_a+1, end_a + 1 #这是因为bert会为文本配上两个特殊标记
                ots, ote = ot.split('#')
                start_o, end_o = int(ots) - 1, int(ote) # 这个因为元数据索引是从0开始
                start_o, end_o = start_o + 1, end_o + 1
                for i in range(start_a, end_a):
                    for j in range(i, end_a):
                        triple_labels[k, i, j] = 1 #aspect
                for i in range(start_o, end_o):
                    for j in range(i, end_o):
                        triple_labels[k, i, j] = 2 #opinion

                s1, s2 = intensity.split('#')
                score1, score2 = float(s1), float(s2)
                for i in range(start_a, end_a):
                    for j in range(start_o, end_o):
                        # ui, uj = min(i, j), max(i,j)
                        score_labels[k, i, j] = score1 / 10
                        if self.use_aspect_class:
                            triple_labels[k, i, j] = label2id[c] + 3 # 因为aspect,opinion,还有无意义词使用了0，1，2三个类别
                        else:
                            triple_labels[k, i, j] = 3 # 因为aspect,opinion,还有无意义词使用了0，1，2三个类别
                        # li, lj = max(i, j), min(i,j)
                        score_labels[k, j, i] = score2 / 10
                # aaa = score_labels.squeeze().numpy().astype(np.int64)
        inputs['triple_labels'] = triple_labels
        inputs['score_labels'] = score_labels
        inputs['texts'] = texts

        return inputs




class CollatorGTSDL:
    def __init__(self, tokenizer, max_seq_length, use_aspect_class=False):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_aspect_class = use_aspect_class

    def __call__(self, batch):

        texts = [x['Sentence'] for x in batch]
        Aspect = [x['Aspect'] for x in batch]
        AspectFromTo = [x['AspectFromTo'] for x in batch]
        Opinion = [x['Opinion'] for x in batch]
        OpinionFromTo = [x['OpinionFromTo'] for x in batch]
        Category = [x['Category'] for x in batch]

        Intensity = [x['Intensity'] for x in batch]


        #  tokenizer对所有文本进行编码
        max_len = max([len(x) for x in texts])
        inputs = {}
        input_ids = self.tokenizer.texts_to_sequences(texts, max_len=max_len)
        inputs['input_ids'] = torch.LongTensor(input_ids)

        length = inputs['input_ids'].shape[1]

        triple_labels = torch.zeros(len(texts), length, length, dtype=torch.int64)
        # aspect_class_labels = torch.zeros(len(texts), length, length, dtype=torch.int64)
        # opinion_index_labels = torch.zeros(len(texts), length, length, dtype=torch.int64)
        score_labels = torch.zeros(len(texts), length, length, dtype=torch.float32)

        for k in range(len(batch)):
            for a, at, o, ot, c, intensity in list(zip(Aspect[k],
                                            AspectFromTo[k],
                                            Opinion[k],
                                            OpinionFromTo[k],
                                            Category[k],
                                            Intensity[k])):
                ats, ate = at.split('#')
                start_a, end_a = int(ats) - 1, int(ate) # 这个因为元数据索引是从0开始
                start_a, end_a = start_a, end_a #这是因为bert会为文本配上两个特殊标记
                ots, ote = ot.split('#')
                start_o, end_o = int(ots) - 1, int(ote) # 这个因为元数据索引是从0开始
                start_o, end_o = start_o, end_o
                for i in range(start_a, end_a):
                    for j in range(i, end_a):
                        triple_labels[k, i, j] = 1 #aspect
                for i in range(start_o, end_o):
                    for j in range(i, end_o):
                        triple_labels[k, i, j] = 2 #opinion

                s1, s2 = intensity.split('#')
                score1, score2 = float(s1), float(s2)
                for i in range(start_a, end_a):
                    for j in range(start_o, end_o):
                        # ui, uj = min(i, j), max(i,j)
                        score_labels[k, i, j] = score1 / 10
                        if self.use_aspect_class:
                            triple_labels[k, i, j] = label2id[c] + 3 # 因为aspect,opinion,还有无意义词使用了0，1，2三个类别
                        else:
                            triple_labels[k, i, j] = 3 # 因为aspect,opinion,还有无意义词使用了0，1，2三个类别
                        # li, lj = max(i, j), min(i,j)
                        score_labels[k, j, i] = score2 / 10
                # aaa = score_labels.squeeze().numpy().astype(np.int64)
        inputs['triple_labels'] = triple_labels
        inputs['score_labels'] = score_labels
        inputs['texts'] = texts

        return inputs



class CollatorEMC:
    def __init__(self, tokenizer, max_seq_length, use_aspect_class=False, use_split_words=False):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_aspect_class = use_aspect_class
        self.use_split_words = use_split_words

    def __call__(self, batch):

        texts = [list(x['Sentence']) for x in batch]

        words = [x['words'] for x in batch]

        Aspect = [x['Aspect'] for x in batch]
        AspectFromTo = [x['AspectFromTo'] for x in batch]
        Opinion = [x['Opinion'] for x in batch]
        OpinionFromTo = [x['OpinionFromTo'] for x in batch]
        Category = [x['Category'] for x in batch]

        Intensity = [x['Intensity'] for x in batch]


        # 使用BERT tokenizer对所有文本进行编码
        inputs = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_seq_length,
                                return_tensors='pt', is_split_into_words=True)
        word_ids = torch.zeros_like(inputs['input_ids'], dtype=torch.int64)
        for i, _word in enumerate(words):
            j = 0
            for w in _word:
                word_ids[i, j+1: j+2+len(w)] = word2id[w.lower()]


        inputs.pop('token_type_ids')
        length = inputs['input_ids'].shape[1]

        pos_matrices, dep_matrices = [], []
        distance_matrices = []

        for data in batch:
            text = data['Sentence']
            words = data['words']
            pos = data['pos']
            dep = data['dep']

            word_indexes = []
            start = 0
            for word in words:
                word_indexes.append((start, start + len(word)))
                start += len(word)

            relations_dict = collections.defaultdict(list)
            for i, (head, label) in enumerate(zip(dep['head'], dep['label'])):
                if head > 0:
                    head_index = head - 1
                    if head_index not in relations_dict:
                        relations_dict[head_index] = []
                    relations_dict[head_index].append((i, label))
                else:
                    relations_dict[i].append((i, 'ROOT'))

            num_tokens = length

            # 初始化特征矩阵
            pos_matrix = torch.full((num_tokens, num_tokens), 0, dtype=torch.long)
            dep_matrix = torch.full((num_tokens, num_tokens), 0, dtype=torch.long)

            # 填充词性和依存关系矩阵
            for i, wi in enumerate(words):
                for j, wj in enumerate(words):
                    si, ei = word_indexes[i]
                    sj, ej = word_indexes[j]
                    # 加1是因为bert特色token [CLS]
                    pos_matrix[si + 1:ei + 1, sj + 1:ej + 1] = pospair2id[f'{pos[i]}-{pos[j]}']  # 假设每个字的词性与头词相同
            # print(pos_matrix)
            for i in relations_dict:
                if relations_dict[i]:
                    for j, label in relations_dict[i]:
                        si, ei = word_indexes[i]
                        sj, ej = word_indexes[j]
                        dep_matrix[si + 1:ei + 1, sj + 1:ej + 1] = deppair2id[label]  # 假设每个字的词性与头词相同
            # print(dep_matrix)
            positions = torch.arange(len(text) + 2)
            dis_matrix = positions[:, np.newaxis] - positions
            distance_matrix = dis_matrix + dis_matrix.max()
            padding_num = num_tokens - dis_matrix.shape[0]
            # 指定填充的大小 (左, 右, 上, 下)
            padding = (0, padding_num, 0, padding_num)
            # 使用 torch.nn.functional.pad 进行填充
            distance_matrix = torch.nn.functional.pad(distance_matrix, padding, mode='constant', value=0)
            pos_matrices.append(pos_matrix)
            dep_matrices.append(dep_matrix)
            distance_matrices.append(distance_matrix)
        pos = torch.stack(pos_matrices)
        dep = torch.stack(dep_matrices)
        dis = torch.stack(distance_matrices)




        triple_labels = torch.zeros(len(texts), length, length, dtype=torch.int64)
        score_labels = torch.zeros(len(texts), length, length, dtype=torch.float32)

        for k in range(len(batch)):
            for a, at, o, ot, c, intensity in list(zip(Aspect[k],
                                            AspectFromTo[k],
                                            Opinion[k],
                                            OpinionFromTo[k],
                                            Category[k],
                                            Intensity[k])):
                ats, ate = at.split('#')
                start_a, end_a = int(ats) - 1, int(ate) # 这个因为元数据索引是从0开始
                start_a, end_a = start_a+1, end_a + 1 #这是因为bert会为文本配上两个特殊标记
                ots, ote = ot.split('#')
                start_o, end_o = int(ots) - 1, int(ote) # 这个因为元数据索引是从0开始
                start_o, end_o = start_o + 1, end_o + 1
                for i in range(start_a, end_a):
                    for j in range(i, end_a):
                        triple_labels[k, i, j] = 1 #aspect
                for i in range(start_o, end_o):
                    for j in range(i, end_o):
                        triple_labels[k, i, j] = 2 #opinion

                s1, s2 = intensity.split('#')
                score1, score2 = float(s1), float(s2)
                for i in range(start_a, end_a):
                    for j in range(start_o, end_o):
                        # ui, uj = min(i, j), max(i,j)
                        score_labels[k, i, j] = score1 / 10
                        if self.use_aspect_class:
                            triple_labels[k, i, j] = label2id[c] + 3 # 因为aspect,opinion,还有无意义词使用了0，1，2三个类别
                        else:
                            triple_labels[k, i, j] = 3 # 因为aspect,opinion,还有无意义词使用了0，1，2三个类别
                        # li, lj = max(i, j), min(i,j)
                        score_labels[k, j, i] = score2 / 10
                # aaa = score_labels.squeeze().numpy().astype(np.int64)

        inputs['pos'] = pos
        inputs['dep'] = dep
        inputs['dis'] = dis
        if self.use_split_words:
            inputs['word_ids'] = word_ids

        inputs['triple_labels'] = triple_labels
        inputs['score_labels'] = score_labels
        inputs['texts'] = texts
        return inputs








def make_plm_dataloader(train_file, valid_file, test_file, tokenizer, model_save_path=None, max_len=128,
                        batch_size=128, use_aspect_class=False):
    train_data = read_data(train_file)
    valid_data = read_data(valid_file)
    test_data = read_data(test_file)

    train_dataset = PLMDataset(train_data)
    valid_dataset = PLMDataset(valid_data)
    test_dataset = PLMDataset(test_data)
    collator = CollatorGTS(tokenizer=tokenizer, max_seq_length=max_len, use_aspect_class=use_aspect_class)

    # 制作dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator)
    return train_dataloader, valid_dataloader, test_dataloader


def make_emc_plm_dataloader(train_file, valid_file, test_file, tokenizer, model_save_path=None, max_len=128,
                        batch_size=128, use_aspect_class=False, use_split_words=False):
    train_data = read_data(train_file)
    valid_data = read_data(valid_file)
    test_data = read_data(test_file)

    train_dataset = PLMDataset(train_data)
    valid_dataset = PLMDataset(valid_data)
    test_dataset = PLMDataset(test_data)
    collator = CollatorEMC(tokenizer=tokenizer, max_seq_length=max_len,
                           use_aspect_class=use_aspect_class, use_split_words=use_split_words)

    # 制作dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator)
    return train_dataloader, valid_dataloader, test_dataloader


def make_dl_dataloader(train_file, valid_file, test_file, model_save_path=None, max_len=128,
                    batch_size=128, char_level=False, use_aspect_class=False):
    train_data = read_data(train_file)
    valid_data = read_data(valid_file)
    test_data = read_data(test_file)
    train_texts = [x['Sentence'] for x in train_data]

    tokenizer = Tokenizer(max_len=max_len, char_level=char_level)
    tokenizer.fit_on_texts(train_texts)

    train_dataset = PLMDataset(train_data)
    valid_dataset = PLMDataset(valid_data)
    test_dataset = PLMDataset(test_data)
    collator = CollatorGTSDL(tokenizer=tokenizer, max_seq_length=max_len, use_aspect_class=use_aspect_class)

    if model_save_path is not None:
        # 保存tokenzier和label_encoder
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        tokenizer_path = os.path.join(model_save_path, 'tokenizer.pkl')
        label_encoder_path = os.path.join(model_save_path, 'label_encoder.pkl')

        tokenizer.save_vocab(os.path.join(model_save_path, 'word2id.json'),
                             os.path.join(model_save_path, 'id2word.json'))

    # 制作dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator)

    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == '__main__':
    # 定义参数
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('D:/plm/bert-base-chinese')
    inputs = tokenizer([['叫', '好'], ['地', '返']])
    # print(inputs)
    test_file = './data/train.json'
    model_save_path = './saved/test/'
    # a, b, c = make_plm_dataloader(test_file, test_file, test_file, tokenizer, model_save_path, max_len=64, batch_size=1)
    # for x in a:
    #     # print(x)
    #     break
    # a, b, c = make_dl_dataloader(test_file, test_file, test_file,  model_save_path, max_len=64, batch_size=1, char_level=True)
    # for x in a:
    #     print(x)
    #     break

    test_file = './pdata/train.json'
    a, b, c = make_emc_plm_dataloader(test_file, test_file, test_file, tokenizer, model_save_path, max_len=64, batch_size=16)
    for x in a:
        print(x['dis'].max())
        # break
