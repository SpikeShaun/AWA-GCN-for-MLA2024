import numpy as np
# 修改函数以找到连续标记为aspect或opinion的单词片段
def find_aspect_opinion_phrases(tags):
    aspect_phrases = []
    opinion_phrases = []

    n = len(tags)
    i = 0
    # 寻找连续的aspect片段
    while i < n:
        if tags[i][i] == 1:
            start = i
            while i < n and tags[i][i] == 1:
                i += 1
            aspect_phrases.append((start, i - 1))
        i += 1

    i = 0
    # 寻找连续的opinion片段
    while i < n:
        if tags[i][i] == 2:
            start = i
            while i < n and tags[i][i] == 2:
                i += 1
            opinion_phrases.append((start, i - 1))
        i += 1
    return aspect_phrases, opinion_phrases



# 检查aspect和opinion之间的情感关系
def extract_4_tuples(tags, scores, aspect_phrases, opinion_phrases):
    tuples4 = []
    for aspect_start, aspect_end in aspect_phrases:
        for opinion_start, opinion_end in opinion_phrases:
            tmp_match_result = tags[aspect_start:aspect_end + 1, opinion_start:opinion_end+1]
            if (tmp_match_result>=3).any() == True:
                # 使用bincount计算频次
                counts = np.bincount(tmp_match_result.reshape(-1))
                # 找到出现次数最多的元素索引
                most_frequent = np.argmax(counts)
                aspect_class = most_frequent

                score1 = scores[aspect_start:aspect_end + 1, opinion_start:opinion_end+1].mean()
                score2 = scores[opinion_start:opinion_end+1, aspect_start:aspect_end + 1].mean()
                score1, score2 = int(round(score1)), int(round(score2))

                tuples4.append(((aspect_start, aspect_end), aspect_class, (opinion_start, opinion_end), (score1, score2)))
    return tuples4


def p_r_f1_metric(real_triple_labels, real_score_labels, pred_triple_labels, pred_score_labels):
    real_score_labels = real_score_labels
    pred_score_labels = pred_score_labels
    n = len(real_triple_labels)
    real_num = 0
    pred_num = 0
    pred_correct_num = 0

    real_num_3 = 0
    pred_num_3 = 0
    pred_correct_num_3 = 0

    for i in range(n):

        aspect_phrases, opinion_phrases = find_aspect_opinion_phrases(real_triple_labels[i])
        real_tuples4 = extract_4_tuples(real_triple_labels[i], real_score_labels[i], aspect_phrases, opinion_phrases)
        real_tuples3 = [list(x) for x in real_tuples4]
        real_tuples3 = [tuple(x[:1]+x[2:]) for x in real_tuples3]

        aspect_phrases, opinion_phrases = find_aspect_opinion_phrases(pred_triple_labels[i])
        pred_tuples4 = extract_4_tuples(pred_triple_labels[i], pred_score_labels[i], aspect_phrases, opinion_phrases)
        pred_tuples3 = [list(x) for x in pred_tuples4]
        pred_tuples3 = [tuple(x[:1]+x[2:]) for x in pred_tuples3]
        
        real_num += len(real_tuples4)
        pred_num += len(pred_tuples4)
        pred_correct_num += len(set(pred_tuples4) & set(real_tuples4))

        real_num_3 += len(real_tuples3)
        pred_num_3 += len(pred_tuples3)
        pred_correct_num_3 += len(set(pred_tuples3) & set(real_tuples3))



    precision = pred_correct_num / (pred_num+1e-9)
    recall = pred_correct_num / (real_num+1e-9)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    precision3 = pred_correct_num_3 / (pred_num_3+1e-9)
    recall3 = pred_correct_num_3 / (real_num_3+1e-9)
    f13 = 2 * precision3 * recall3 / (precision3 + recall3) if (precision3 + recall3) != 0 else 0



    return {'precision': precision, 'recall': recall, 'f1': f1,
            'precision3': precision3, 'recall3': recall3, 'f13': f13,}

