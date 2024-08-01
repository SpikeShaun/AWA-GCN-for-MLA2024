from trainers import DLTrainer
from models.dl_models import GTSBiLSTM, GTSCNN, GTSTransformer
from dataset import make_dl_dataloader
import json
import torch
import os
import numpy as np

seed = 56
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

# 定义参数
train_file = './data/train.json'
valid_file = './data/test.json'
test_file = './data/test.json'

# 模型相关
save_path = './saved/'
use_aspect_class = True
if use_aspect_class:
    num_labels = 16
else:
    num_labels = 4


# cnn使用这里的配置
model_config = {
    "num_labels": num_labels,  # 这里取决于数据中有多少类别, 可以手动指定, 如果不手动指定的话就根据数据中的类别数目自动设定
    "vocab_size": 10000,
    "embedding_dim": 256,
    "num_filters": 256,
    "kernel_sizes": [2,3,4]
}
#
# # lstm 使用这里的配置
# model_config = {
#     "num_labels": num_labels,  # 这里取决于数据中有多少类别, 可以手动指定, 如果不手动指定的话就根据数据中的类别数目自动设定
#     "vocab_size": 10000,
#     "embedding_dim": 256,
#     "hidden_size": 256,
#     "num_layers": 2
# }

# # transformer 使用这里的配置
# model_config = {
#     "num_labels": num_labels,  # 这里取决于数据中有多少类别, 可以手动指定, 如果不手动指定的话就根据数据中的类别数目自动设定
#     "vocab_size": 10000,
#     "embedding_dim": 256,
#     "hidden_size": 256,
#     "num_layers": 4
# }

MODEL = GTSCNN
model = MODEL(**model_config)


model_name = type(model).__name__
model_save_path = os.path.join(save_path,model_name)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

json.dump(model_config, open(os.path.join(model_save_path, 'config.json'), 'w'))

# 训练相关
lr = 0.0001
num_epochs = 100
batch_size = 64
max_length = 64
criterion = torch.nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dataloader, valid_dataloader, test_dataloader = make_dl_dataloader(
    train_file, valid_file, test_file, model_save_path=model_save_path,
    max_len=max_length, batch_size=batch_size, char_level=True, use_aspect_class=use_aspect_class)


trainer = DLTrainer(model, train_dataloader, valid_dataloader,
                 lr=lr, num_epochs=num_epochs, batch_size=batch_size, save_path=save_path,
                 model_name=model_name, monitor='f1', average='binary',
                 criterion=criterion, device=device)
# trainer.training()
#
# # 评估
model_save_path = os.path.join(save_path, model_name)
model_config = json.load(open(os.path.join(model_save_path, 'config.json')))
model.load_state_dict(torch.load(os.path.join(model_save_path, 'weights.pth'), map_location=torch.device('cpu')))
model.to(device=device)
trainer.model = model
test_metrics = trainer.evaluate(test_dataloader, criterion=criterion)
print(f'test_metrics: {test_metrics}')
json.dump(test_metrics, open(os.path.join('model_metrics', f'{model_name}.json'),'w'))
