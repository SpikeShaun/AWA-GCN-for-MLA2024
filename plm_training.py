from trainers import PLMTrainer
from models.plm_models import GTSBert
from dataset import make_plm_dataloader
from transformers import BertTokenizerFast
import torch
import os
import numpy as np
import random
import json

# 设置随机库的种子
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个CUDA设备
np.random.seed(seed)
random.seed(seed)
# 为了确保CUDA的确定性行为，可以设置以下两个配置，但这可能会牺牲一些性能
# 注意：这可能会影响性能
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 定义参数
train_file = './data/train.json'
valid_file = './data/test.json'
test_file = './data/test.json'

# 模型相关
save_path = './saved/'

pretrained_path = 'D:/plm/bert-base-chinese'
use_aspect_class = True
if use_aspect_class:
    num_labels = 16
else:
    num_labels = 4
model = GTSBert.from_pretrained(pretrained_path, num_labels=num_labels)
model_name = type(model).__name__
model_save_path = os.path.join(save_path,model_name)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)


# 训练相关
lr = 2e-5
num_epochs = 50
batch_size = 64
max_length = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizerFast.from_pretrained(pretrained_path)

train_dataloader, valid_dataloader, test_dataloader = make_plm_dataloader(
    train_file, valid_file, test_file, tokenizer=tokenizer, model_save_path=model_save_path,
    max_len=max_length, batch_size=batch_size, use_aspect_class=use_aspect_class)


trainer = PLMTrainer(model, train_dataloader, valid_dataloader,
                 lr=lr, num_epochs=num_epochs, batch_size=batch_size, save_path=save_path,
                 model_name=model_name, monitor='f1', average='binary',
                 device=device)
# trainer.training()

# 评估
model_save_path = os.path.join(save_path, model_name)
trainer.model = GTSBert.from_pretrained(model_save_path)
trainer.model.to(device)
test_metrics = trainer.evaluate(test_dataloader, is_test=True)
print(f'test_metrics: {test_metrics}')
# test_metrics['confusion_matrix'] = test_metrics['confusion_matrix'].tolist()
json.dump(test_metrics, open(os.path.join('model_metrics', f'{model_name}.json'),'w'))
