import json
import re
import collections
import random

p1='./dimABSA_TrainingSetValidationSet/dimABSA_TrainingSet&ValidationSet/dimABSA_TrainingSet1/SIGHAN2024_dimABSA_TrainingSet1_Simplified.json'
p2='./dimABSA_TrainingSetValidationSet/dimABSA_TrainingSet&ValidationSet/dimABSA_TrainingSet2/SIGHAN2024_dimABSA_TrainingSet2_Simplified.json'

data1 = json.load(open(p1, encoding='utf-8'))
data2 = json.load(open(p2, encoding='utf-8'))
data12 = data1 + data2
data = []
for item in data12:
    Aspect = item['Aspect']
    AspectFromTo = item['AspectFromTo']
    Opinion = item['Opinion']
    OpinionFromTo = item['OpinionFromTo']
    Sentence = item['Sentence']
    bad_data = False
    for a, at, o, ot in list(zip(Aspect, AspectFromTo, Opinion, OpinionFromTo)):

        ats, ate = at.split('#')
        ats, ate = int(ats)-1, int(ate)
        if Sentence[ats:ate] != a and a !='NULL':
            print(Sentence, a)
            bad_data = True

        ots, ote = ot.split('#')
        ots, ote = int(ots)-1, int(ote)
        if Sentence[ots:ote] != o and o!='NULL':
            print(Sentence, o)
            bad_data = True
    if not bad_data:
        data.append(item)

random.seed(3)
random.shuffle(data)


test_size = 0.2
train_size = 1-test_size
train_num = int(len(data) * train_size)
train_data = data[:train_num]
test_data = data[train_num:]
json.dump(data, open('./data/total.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
json.dump(train_data, open('./data/train.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
json.dump(test_data, open('./data/test.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

classes = [y for x in data for y in x['Category']]

count = collections.Counter(classes)
label2id = {k:i+1 for i, k in enumerate(count)}
label2id['NULL'] = 0

json.dump(label2id, open('./data/label2id.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)