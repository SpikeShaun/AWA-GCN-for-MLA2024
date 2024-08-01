
from ltp import LTP
import json

# 初始化 LTP 模型
def linguist_parse(data, batch_size=64):
    i = 0
    parse_data = []
    while i < len(data):
        batch_data = data[i:i+batch_size]
        sentences = [x['Sentence'] for x in batch_data]
        output = ltp.pipeline(sentences, tasks=["cws", "pos","dep",])
        for j in range(len(batch_data)):
            batch_data[j]['words'] = output.cws[j]
            batch_data[j]['pos'] = output.pos[j]
            batch_data[j]['dep'] = output.dep[j]
            parse_data.append(batch_data[j])
        i += batch_size
        print(f'process for {i} data..........')
    return parse_data

if __name__ == '__main__':
    ltp = LTP(pretrained_model_name_or_path='./LTP/base')

    # data = json.load(open('./data/test.json', encoding='utf-8'))
    # pdata1 = linguist_parse(data)
    # json.dump(pdata1, open('./pdata/test.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    #
    # data = json.load(open('./data/train.json', encoding='utf-8'))
    # pdata2 = linguist_parse(data)
    # json.dump(pdata2, open('./pdata/train.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    pdata1 = json.load(open('./pdata/test.json', encoding='utf-8'), )
    pdata2 = json.load(open('./pdata/train.json', encoding='utf-8'), )
    pdata = pdata1 + pdata2

    # pos = [y for x in pdata for y in x['pos']]
    # dep = [y for x in pdata for y in x['dep']['label']]
    # pos = sorted(set(pos))
    # dep = sorted(set(dep))
    # print(len(pos), pos)
    # print(len(dep), dep)
    #
    # pospair = [f'{pos[i]}-{pos[j]}' for i in range(len(pos)) for j in range(len(pos))]
    # deppair = dep
    #
    # pospair2id = {p:i+1 for i, p in enumerate(pospair)}
    # deppair2id = {p:i+2 for i, p in enumerate(deppair)}
    # deppair2id['ROOT'] = 1
    # print(pospair2id)
    # print(deppair2id)
    #
    # json.dump(pospair2id, open('./pdata/pospair2id.json', 'w', encoding='utf-8'), indent=1)
    # json.dump(deppair2id, open('./pdata/deppair2id.json', 'w', encoding='utf-8'), indent=1)

    word2id = {'<pad>':0, '<unknown>':1}
    words = [y.lower() for x in pdata for y in x['words']]
    words = sorted(set(words))
    for i, w in enumerate(words):
        word2id[w] = i + 2
    json.dump(word2id, open('./pdata/word2id.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)










