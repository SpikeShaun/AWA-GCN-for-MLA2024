import torch
from models.dl_models import BiGRU,BiLSTM,TextCNN
from models.plm_models import ALBertLstmCNN
import json
import os
from dataset import make_dataloader, make_plm_dataloader
from transformers import BertTokenizerFast
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from metrics import cal_acc_precision_recall_f1
import pandas as pd
albert = ALBertLstmCNN.from_pretrained('./saved/ALBertLstmCNN')

model_config = json.load(open(os.path.join('./saved/BiLSTM', 'config.json')))
bilstm = BiLSTM(**model_config)
bilstm.load_state_dict(torch.load(os.path.join('./saved/BiLSTM', 'weights.pth')))

model_config = json.load(open(os.path.join('./saved/BiGRU', 'config.json')))
bigru = BiGRU(**model_config)
bigru.load_state_dict(torch.load(os.path.join('./saved/BiGRU', 'weights.pth')))

model_config = json.load(open(os.path.join('./saved/TextCNN', 'config.json')))
textcnn = TextCNN(**model_config)
textcnn.load_state_dict(torch.load(os.path.join('./saved/TextCNN', 'weights.pth')))

# 定义参数
train_file = './data/train.csv'
valid_file = './data/valid.csv'
test_file = './data/test.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
max_length = 128
pretrained_path = './albert-base-chinese'
label_list = []
plabel_list = []
prob_list = []
for model in [albert, bilstm, bigru, textcnn]:
    model.to(device)
    model.eval()
    real_labels = []
    pred_labels = []
    pred_probs = []
    if isinstance(model, ALBertLstmCNN):
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_path)
        train_dataloader, valid_dataloader, test_dataloader = make_plm_dataloader(
            train_file, valid_file, test_file, tokenizer=tokenizer, max_len=max_length,
            batch_size=batch_size)

        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                # 将数据和标签移动到指定设备上
                batch = {k: batch[k].to(device) for k in batch}
                output = model(**batch)
                loss = output.loss
                _pred_labels = torch.argmax(output.logits, dim=-1)
                _pred_probs = torch.softmax(output.logits, dim=-1)[:,1]
                real_labels += batch['labels'].cpu().numpy().tolist()
                pred_labels += _pred_labels.cpu().numpy().tolist()
                pred_probs += _pred_probs.cpu().numpy().tolist()

    else:

        train_dataloader, valid_dataloader, test_dataloader = make_dataloader(
            train_file, valid_file, test_file, model_save_path=None, max_len=max_length,
            batch_size=batch_size, char_level=True)
        with torch.no_grad():
            for inp, tgt in tqdm(test_dataloader):
                # 将数据和标签移动到指定设备上
                inp = inp.to(device)
                tgt = tgt.to(device)
                output = model(inp)
                _pred_labels = torch.argmax(output, dim=-1)
                _pred_probs = torch.softmax(output, dim=-1)[:,1]
                real_labels += tgt.cpu().numpy().tolist()
                pred_labels += _pred_labels.cpu().numpy().tolist()
                pred_probs += _pred_probs.cpu().numpy().tolist()

    plabel_list.append(pred_labels)
    prob_list.append(pred_probs)
    label_list.append(real_labels)


from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.xticks([0,1], ['negative', 'positive'])
    plt.yticks([0,1], ['negative', 'positive'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# 画混淆矩阵
# 假设 label_list 是你所有模型的真实标签列表
# plabel_list 是一个列表的列表，其中每个子列表包含了对应模型的预测标签
model_names = ['ALBertLstmCnn', 'BiLSTM', 'BiGRU', 'TextCNN']

# 创建 1x4 的图
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

for i, (plabels, model_name) in enumerate(zip(plabel_list, model_names)):
    # 计算混淆矩阵
    cm = confusion_matrix(label_list[i], plabels)
    plt.subplot(1, 4, i+1)
    plot_confusion_matrix(cm, title=model_name)

plt.savefig('./static/images/confusion_matrix.jpeg', dpi=300)
plt.show()


# 画指标图
# 假设 label_list 是你所有模型的真实标签列表
# plabel_list 是一个列表的列表，其中每个子列表包含了对应模型的预测标签
# model_names = ['ALBertLstmCnn', 'BiLSTM', 'BiGRU', 'TextCNN']

# 创建 1x4 的图
fig, ax = plt.subplots(1, 4, figsize=(20, 4))

m_list = []
for i, (plabels, model_name) in enumerate(zip(plabel_list, model_names)):
    # 计算混淆矩阵
    m = cal_acc_precision_recall_f1(label_list[i], plabels, average='binary')
    m.pop('auc')
    m_list.append(m)
dfm = pd.DataFrame(m_list, index=model_names)
print(dfm)
colors = ['lightsalmon', 'lightgreen', 'lightblue', 'lightpink']#, 'wheat', 'lavender']

for i, c in enumerate(dfm.columns):
    dfm[c].plot(kind='bar', ax=ax[i], color=colors, rot=0)
    ax[i].set_ylim([0.85, 0.948])
    ax[i].set_ylabel(c)
plt.tight_layout()
plt.savefig('./static/images/metrics.jpeg', dpi=300)
plt.show()


# 准备真实标签和模型预测概率的列表
# 假设每个模型的预测概率和真实标签分别存储在 y_probs 和 y_true_lists 中
# model_names = ['ALBert', 'BiLSTM', 'BiGRU', 'TextCNN']

# 创建一个图形
plt.figure(figsize=(8, 6))
# 循环遍历每个模型
for i in range(len(prob_list)):
    # 计算 ROC 曲线
    fpr, tpr, _ = roc_curve(label_list[i], prob_list[i])
    roc_auc = roc_auc_score(label_list[i], prob_list[i])

    # 绘制 ROC 曲线
    plt.plot(fpr, tpr, label=model_names[i]+' (AUC = {:.4f})'.format(roc_auc))

# 添加标签和图例
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('static/images/roc.jpeg', dpi=300)
# 显示图形
plt.show()

# 画混淆矩阵图

