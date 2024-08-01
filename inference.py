from trainers import PLMTrainer
from models.plm_models import EMCGCN_ABFF_Word, EMCGCN_ABFF_Word_ATT
from dataset import make_emc_plm_dataloader
from transformers import BertTokenizerFast
import torch
import os
import numpy as np
import random
from tqdm import tqdm
import json
from metrics import extract_4_tuples
from metrics import find_aspect_opinion_phrases
# 定义参数
label2id = json.load(open('./data/label2id.json', encoding='utf-8'))
id2label = {v+3: k for k, v in label2id.items()}

test_file = './pdata/test.json'

# 模型相关
save_path = './saved/'
use_aspect_class = True
if use_aspect_class:
    num_labels = 16
else:
    num_labels = 4
MODEL = EMCGCN_ABFF_Word_ATT
model_name = MODEL.__name__
model_save_path = os.path.join(save_path,model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MODEL.from_pretrained(model_save_path)
model.to(device)

# 训练相关
batch_size = 8
max_length = 96
pretrained_path = 'D:/plm/bert-base-chinese'
tokenizer = BertTokenizerFast.from_pretrained(pretrained_path)

_, _, dataloader = make_emc_plm_dataloader(
    test_file, test_file, test_file, tokenizer=tokenizer, model_save_path=model_save_path,
    max_len=max_length, batch_size=batch_size, use_aspect_class=use_aspect_class, use_split_words=True)

# 评估
model.eval()
valid_loss = 0
texts = []
real_labels = []
real_scores = []
pred_labels = []
pred_scores = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        _texts = batch.pop('texts')
        # 将数据和标签移动到指定设备上
        batch = {k: batch[k].to(device) for k in batch}
        output = model(**batch)
        loss = output.loss
        valid_loss += loss.item()
        _pred_labels = torch.argmax(output.logits, dim=-1).cpu().numpy()
        _pred_scores = output.scores.cpu().detach().numpy() * 10
        real_labels += list(batch['triple_labels'].cpu().numpy())
        real_scores += list(batch['score_labels'].cpu().numpy() * 10)
        pred_labels += list(_pred_labels)
        pred_scores += list(_pred_scores)
        texts += _texts
        # break


def decode(text, tuples4):

    item = []
    for tuple in tuples4:

        a_s, a_e = tuple[0]
        aspect = ''.join(text[a_s-1:a_e])
        category = tuple[1]
        if category < 3:
            continue
        category = id2label[category]
        o_s, o_e = tuple[2]
        opinion = ''.join(text[o_s-1:o_e])
        score = tuple[3]
        score = ','.join(list(map(str, score)))

        item.append({'aspect': aspect, 'category':category, 'opinion':opinion, 'score':score})
    return item

result = []
n = len(texts)
for i in range(n):
    text = texts[i]

    aspect_phrases, opinion_phrases = find_aspect_opinion_phrases(real_labels[i])
    real_tuples4 = extract_4_tuples(real_labels[i], real_scores[i], aspect_phrases, opinion_phrases)
    
    aspect_phrases, opinion_phrases = find_aspect_opinion_phrases(pred_labels[i])
    pred_tuples4 = extract_4_tuples(pred_labels[i], pred_scores[i], aspect_phrases, opinion_phrases)

    items = {
        'text': ''.join(text),
        'real_4_tuples': decode(text, real_tuples4),
        'pred_4_tuples': decode(text, pred_tuples4)
    }
    result.append(items)

json.dump(result, open('./pdata/result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)









