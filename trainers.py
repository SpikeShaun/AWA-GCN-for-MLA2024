import os
import collections
import torch
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from metrics import p_r_f1_metric
import math
import os
import logging
import json
import numpy as np
import datetime

# 创建一个Formatter对象，包含时间信息
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 配置日志记录器并设置Formatter
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DLTrainer:
    def __init__(self, model, train_dataloader, valid_dataloader,
                 lr=0.001, num_epochs=10, batch_size=128, save_path='./save/',
                 model_name='model', monitor='loss', average='binary',
                 criterion=None, device=None):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.save_path = save_path
        self.model = model
        self.model.to(device)
        self.model_name = model_name
        self.device = device
        self.best_metric = -float('inf')
        self.monitor = monitor
        self.average = average
        self.model_save_path = os.path.join(save_path, model_name)

    def train(self, dataloader, optimizer, criterion=None, scheduler=None):
        self.model.train()
        train_loss = 0
        real_labels = []
        real_scores = []
        pred_labels = []
        pred_scores = []
        for batch in tqdm(dataloader):
            batch.pop('texts')

            # 将梯度清零
            optimizer.zero_grad()
            # 将数据和标签移动到指定设备上
            batch = {k: batch[k].to(self.device) for k in batch}
            # 前向传播
            output = self.model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            train_loss += loss.item()
            _pred_labels = torch.argmax(output.logits, dim=-1).cpu().numpy()
            _pred_scores = output.scores.cpu().detach().numpy()*10
            real_labels += list(batch['triple_labels'].cpu().numpy())
            real_scores += list(batch['score_labels'].cpu().numpy()*10)
            pred_labels += list(_pred_labels)
            pred_scores += list(_pred_scores)

        # metric = p_r_f1_metric(real_labels, real_scores, pred_labels, pred_scores)
        metric = {}
        metric['loss'] = train_loss / len(dataloader)
        return metric

    def evaluate(self, dataloader, criterion=None, is_test=False):
        self.model.eval()
        valid_loss = 0
        real_labels = []
        real_scores = []
        pred_labels = []
        pred_scores = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch.pop('texts')

                # 将数据和标签移动到指定设备上
                batch = {k: batch[k].to(self.device) for k in batch}
                output = self.model(**batch)
                loss = output.loss
                valid_loss += loss.item()
                _pred_labels = torch.argmax(output.logits, dim=-1).cpu().numpy()
                _pred_scores = output.scores.cpu().detach().numpy() * 10
                real_labels += list(batch['triple_labels'].cpu().numpy())
                real_scores += list(batch['score_labels'].cpu().numpy() * 10)
                pred_labels += list(_pred_labels)
                pred_scores += list(_pred_scores)
        metric = p_r_f1_metric(real_labels, real_scores, pred_labels, real_scores)
        metric['loss'] = valid_loss / len(dataloader)
        return metric

    def save_best_model(self, save_path, curr_metric):
        if self.monitor == 'loss':
            curr_metric = -curr_metric
        if self.best_metric <= curr_metric:
            model_file_path = os.path.join(save_path, 'weights.pth')
            self.best_metric = curr_metric
            torch.save(self.model.state_dict(), model_file_path)

    def training(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        total_steps = len(self.train_dataloader) * self.num_epochs
        warmup_steps = math.ceil(total_steps * 0.05)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps, last_epoch=-1)
        train_history = collections.defaultdict(list)
        valid_history = collections.defaultdict(list)
        for epoch in range(1, self.num_epochs + 1):
            train_metric = self.train(self.train_dataloader, optimizer, criterion=self.criterion, scheduler=scheduler)
            valid_metric = self.evaluate(self.valid_dataloader, criterion=self.criterion)
            print(f'Epoch {epoch}: Train metric={train_metric}\nvalid metric={valid_metric}')
            logger.info(f'Epoch {epoch}: Train metric={train_metric}\nvalid metric={valid_metric}')

            for k, v in train_metric.items():
                train_history[k].append(v)
            for k, v in valid_metric.items():
                valid_history[k].append(v)
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)
            self.save_best_model(self.model_save_path, curr_metric=valid_metric[self.monitor])
        history = {'train_history': train_history, 'valid_history': valid_history}
        json.dump(history,
                  open(os.path.join(self.model_save_path, f'{self.model_name}_history.json'), 'w', encoding='utf-8'))
        return history




class PLMTrainer:
    def __init__(self, model, train_dataloader, valid_dataloader,
                 lr=0.001, num_epochs=10, batch_size=128, save_path='./save/',
                 model_name='model', monitor='loss', average='binary',
                 criterion=None, device=None):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.save_path = save_path
        self.model = model
        self.model.to(device)
        self.model_name = model_name
        self.device = device
        self.best_metric = -float('inf')
        self.monitor = monitor
        self.average = average
        self.model_save_path = os.path.join(save_path, model_name)

    def train(self, dataloader, optimizer, criterion=None, scheduler=None):
        self.model.train()
        train_loss = 0
        real_labels = []
        real_scores = []
        pred_labels = []
        pred_scores = []
        for batch in tqdm(dataloader):
            batch.pop('texts')
            # 将梯度清零
            optimizer.zero_grad()
            # 将数据和标签移动到指定设备上
            batch = {k: batch[k].to(self.device) for k in batch}
            # 前向传播
            output = self.model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            train_loss += loss.item()
            _pred_labels = torch.argmax(output.logits, dim=-1).cpu().numpy()
            _pred_scores = output.scores.cpu().detach().numpy() * 10
            real_labels += list(batch['triple_labels'].cpu().numpy())
            real_scores += list(batch['score_labels'].cpu().numpy() * 10)
            pred_labels += list(_pred_labels)
            pred_scores += list(_pred_scores)

        # metric = p_r_f1_metric(real_labels, real_scores, pred_labels, pred_scores)
        metric = {}
        metric['loss'] = train_loss / len(dataloader)
        return metric

    def evaluate(self, dataloader, criterion=None, is_test=False):
        self.model.eval()
        valid_loss = 0
        real_labels = []
        real_scores = []
        pred_labels = []
        pred_scores = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch.pop('texts')

                # 将数据和标签移动到指定设备上
                batch = {k: batch[k].to(self.device) for k in batch}
                output = self.model(**batch)
                loss = output.loss
                valid_loss += loss.item()
                _pred_labels = torch.argmax(output.logits, dim=-1).cpu().numpy()
                _pred_scores = output.scores.cpu().detach().numpy() * 10
                real_labels += list(batch['triple_labels'].cpu().numpy())
                real_scores += list(batch['score_labels'].cpu().numpy() * 10)
                pred_labels += list(_pred_labels)
                pred_scores += list(_pred_scores)
        metric = p_r_f1_metric(real_labels, real_scores, pred_labels, real_scores)
        metric['loss'] = valid_loss / len(dataloader)
        return metric

    def save_best_model(self, save_path, curr_metric):
        if self.monitor == 'loss':
            curr_metric = -curr_metric
        if self.best_metric <= curr_metric:
            self.best_metric = curr_metric
            self.model.save_pretrained(save_path)

    def training(self):

        # # 分离BERT参数和其他参数
        # bert_params = []
        # task_params = []
        # for name, param in self.model.named_parameters():
        #     if 'bert' in name:
        #         bert_params.append(param)
        #     else:
        #         task_params.append(param)
        #
        # # 创建参数组，为不同部分设置不同的学习率
        # param_groups = [
        #     {'params': bert_params, 'lr': self.lr},
        #     {'params': task_params, 'lr': self.lr * 10}
        # ]
        # optimizer = AdamW(param_groups, lr=self.lr, weight_decay=1e-4)
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        total_steps = len(self.train_dataloader) * self.num_epochs
        warmup_steps = math.ceil(total_steps * 0.05)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps, last_epoch=-1)
        train_history = collections.defaultdict(list)
        valid_history = collections.defaultdict(list)
        for epoch in range(1, self.num_epochs + 1):
            train_metric = self.train(self.train_dataloader, optimizer, criterion=self.criterion, scheduler=scheduler)
            valid_metric = self.evaluate(self.valid_dataloader, criterion=self.criterion)
            # print(f'Epoch {epoch}: Train metric={train_metric}\nvalid metric={valid_metric}')
            logger.info(f'Epoch {epoch}: Train metric={train_metric}\nvalid metric={valid_metric}')

            for k, v in train_metric.items():
                train_history[k].append(v)
            for k, v in valid_metric.items():
                valid_history[k].append(v)
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)
            self.save_best_model(self.model_save_path, curr_metric=valid_metric[self.monitor])
        history = {'train_history': train_history, 'valid_history': valid_history}
        json.dump(history,
                  open(os.path.join(self.model_save_path, f'{self.model_name}_history.json'), 'w', encoding='utf-8'))
        return history


