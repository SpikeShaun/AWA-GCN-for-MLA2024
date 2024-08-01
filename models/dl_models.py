import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GTSBiLSTM(nn.Module):

    # 定义初始化函数，接受以下参数：
    # num_labels: 分类任务中的类别数量
    # vocab_size: 用于将单词转换为向量表示的词汇表大小，默认为10000
    # embedding_dim: 词向量的维度，默认为256
    # hidden_size: LSTM层中隐藏状态的维度，默认为256
    # num_layers: LSTM层的数量，默认为2
    def __init__(self, num_labels, vocab_size=10000, embedding_dim=256, hidden_size=256, num_layers=2, dropout_rate=0.2):
        # 调用父类初始化函数
        super(GTSBiLSTM, self).__init__()

        # 定义一个嵌入层，用于将单词转换为向量表示
        # 参数vocab_size：词汇表大小
        # 参数embedding_dim：词向量的维度
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # 定义一个双向LSTM层
        # 参数input_size：输入的特征维度，即词向量的维度
        # 参数hidden_size：LSTM层中隐藏状态的维度的一半（因为是双向LSTM）
        # 参数bidirectional：是否使用双向LSTM
        # 参数num_layers：LSTM层的数量
        # 参数batch_first：是否使用batch_first格式，即第一个维度是batch size
        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size // 2,
                              bidirectional=True, num_layers=num_layers, batch_first=True)

        # 定义一个dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 定义一个线性层，用于将LSTM层的输出转换为类别概率
        # 参数in_features：线性层的输入维度，即LSTM层输出的维度
        # 参数out_features：线性层的输出维度，即类别数量
        self.triple_linear = nn.Linear(hidden_size, num_labels)

        self.score_linear = nn.Linear(hidden_size, 1)
    # 定义前向传递函数
    # 参数inputs：输入的数据，即单词序列的整数表示
    def forward(self, input_ids,
                triple_labels=None,
                score_labels=None,):
        # 将输入数据传入嵌入层，得到词向量表示
        embedding = self.embedding(input_ids)

        # 将词向量表示传入LSTM层，得到LSTM层的输出和最终的隐藏状态
        # output是LSTM层的输出，ht和ct是最终的隐藏状态
        output, (ht, ct) = self.bilstm(embedding)

        # 使用 einsum 来生成 word-pair representations
        # 'bik,bjk->bijK' 这个等式意味着对于每个批次中的每对序列位置，我们复制相应的hidden state
        word_pair_feature = torch.einsum('bik,bjk->bijk', [output, output])
        triple_logits = self.triple_linear(word_pair_feature)
        scores = self.score_linear(word_pair_feature).squeeze(-1)
        loss = None
        if triple_labels is not None:
            loss = self.loss_compute(triple_logits, scores,
                                     triple_labels,
                                     score_labels
                                     )
        output = ModelOutput(
            loss=loss,
            logits=triple_logits,
            scores=scores,
        )
        return output

    def loss_compute(self, triple_logits, scores,
                     triple_labels,
                     score_labels
                     ):
        # 创建上三角掩码
        batch_size = triple_logits.shape[0]
        seq_len = triple_logits.shape[1]
        mask_up = torch.triu(torch.ones(seq_len, seq_len), diagonal=0).bool()

        # 扩展掩码到 batch_size
        mask_up = mask_up.unsqueeze(0).expand(batch_size, -1, -1)
        loss_fn1 = nn.CrossEntropyLoss()

        # 使用掩码过滤 logits 和 labels
        num_classes = triple_logits.shape[-1]

        mask_triple_logits = triple_logits[mask_up.unsqueeze(-1).expand_as(triple_logits)].view(-1, num_classes)
        mask_triple_labels = triple_labels[mask_up].view(-1)
        triple_loss = loss_fn1(mask_triple_logits, mask_triple_labels)

        # 分数loss
        loss_fn2 = nn.MSELoss()
        score_loss = loss_fn2(scores, score_labels)

        loss = triple_loss + score_loss
        return loss



class GTSTransformer(nn.Module):

    # 定义初始化函数，接受以下参数：
    # num_labels: 分类任务中的类别数量
    # vocab_size: 用于将单词转换为向量表示的词汇表大小，默认为10000
    # embedding_dim: 词向量的维度，默认为256
    # hidden_size: LSTM层中隐藏状态的维度，默认为256
    # num_layers: LSTM层的数量，默认为2
    def __init__(self, num_labels, vocab_size=10000, embedding_dim=256, hidden_size=256, num_layers=2, dropout_rate=0.2):
        # 调用父类初始化函数
        super(GTSTransformer, self).__init__()

        # 定义一个嵌入层，用于将单词转换为向量表示
        # 参数vocab_size：词汇表大小
        # 参数embedding_dim：词向量的维度
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # 定义一个双向LSTM层
        # 参数input_size：输入的特征维度，即词向量的维度
        # 参数hidden_size：LSTM层中隐藏状态的维度的一半（因为是双向LSTM）
        # 参数bidirectional：是否使用双向LSTM
        # 参数num_layers：LSTM层的数量
        # 参数batch_first：是否使用batch_first格式，即第一个维度是batch size
        self.transformer = TransformerEncoder(TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=1024,
            batch_first=True,
        ), num_layers=num_layers)

        # 定义一个dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 定义一个线性层，用于将LSTM层的输出转换为类别概率
        # 参数in_features：线性层的输入维度，即LSTM层输出的维度
        # 参数out_features：线性层的输出维度，即类别数量
        self.triple_linear = nn.Linear(hidden_size, num_labels)

        self.score_linear = nn.Linear(hidden_size, 1)
    # 定义前向传递函数
    # 参数inputs：输入的数据，即单词序列的整数表示
    def forward(self, input_ids,
                triple_labels=None,
                score_labels=None,):
        # 将输入数据传入嵌入层，得到词向量表示
        embedding = self.embedding(input_ids)

        # 将词向量表示传入LSTM层，得到LSTM层的输出和最终的隐藏状态
        # output是LSTM层的输出，ht和ct是最终的隐藏状态
        output = self.transformer(embedding)

        # 使用 einsum 来生成 word-pair representations
        # 'bik,bjk->bijK' 这个等式意味着对于每个批次中的每对序列位置，我们复制相应的hidden state
        word_pair_feature = torch.einsum('bik,bjk->bijk', [output, output])
        triple_logits = self.triple_linear(word_pair_feature)
        scores = self.score_linear(word_pair_feature).squeeze(-1)
        loss = None
        if triple_labels is not None:
            loss = self.loss_compute(triple_logits, scores,
                                     triple_labels,
                                     score_labels
                                     )
        output = ModelOutput(
            loss=loss,
            logits=triple_logits,
            scores=scores,
        )
        return output

    def loss_compute(self, triple_logits, scores,
                     triple_labels,
                     score_labels
                     ):
        # 创建上三角掩码
        batch_size = triple_logits.shape[0]
        seq_len = triple_logits.shape[1]
        mask_up = torch.triu(torch.ones(seq_len, seq_len), diagonal=0).bool()

        # 扩展掩码到 batch_size
        mask_up = mask_up.unsqueeze(0).expand(batch_size, -1, -1)
        loss_fn1 = nn.CrossEntropyLoss()

        # 使用掩码过滤 logits 和 labels
        num_classes = triple_logits.shape[-1]

        mask_triple_logits = triple_logits[mask_up.unsqueeze(-1).expand_as(triple_logits)].view(-1, num_classes)
        mask_triple_labels = triple_labels[mask_up].view(-1)
        triple_loss = loss_fn1(mask_triple_logits, mask_triple_labels)

        # 分数loss
        loss_fn2 = nn.MSELoss()
        score_loss = loss_fn2(scores, score_labels)

        loss = triple_loss + score_loss
        return loss

class GTSCNN(nn.Module):

    # 定义初始化函数，接受以下参数：
    # num_labels: 分类任务中的类别数量
    # vocab_size: 用于将单词转换为向量表示的词汇表大小，默认为10000
    # embedding_dim: 词向量的维度，默认为256
    # hidden_size: LSTM层中隐藏状态的维度，默认为256
    # num_layers: LSTM层的数量，默认为2
    def __init__(self, num_labels, vocab_size=10000, embedding_dim=256, num_filters=256,
                 kernel_sizes=[2,3,4], dropout_rate=0.2):
        # 调用父类初始化函数
        super(GTSCNN, self).__init__()

        # 定义一个嵌入层，用于将单词转换为向量表示
        # 参数vocab_size：词汇表大小
        # 参数embedding_dim：词向量的维度
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # 定义一个双向LSTM层
        # 参数input_size：输入的特征维度，即词向量的维度
        # 参数hidden_size：LSTM层中隐藏状态的维度的一半（因为是双向LSTM）
        # 参数bidirectional：是否使用双向LSTM
        # 参数num_layers：LSTM层的数量
        # 参数batch_first：是否使用batch_first格式，即第一个维度是batch size
        self.cnns = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding='same') for k in kernel_sizes])

        # 定义一个dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 定义一个线性层，用于将LSTM层的输出转换为类别概率
        # 参数in_features：线性层的输入维度，即LSTM层输出的维度
        # 参数out_features：线性层的输出维度，即类别数量
        self.triple_linear = nn.Linear(num_filters*len(kernel_sizes), num_labels)

        self.score_linear = nn.Linear(num_filters*len(kernel_sizes), 1)
    # 定义前向传递函数
    # 参数inputs：输入的数据，即单词序列的整数表示
    def forward(self, input_ids,
                triple_labels=None,
                score_labels=None,):
        # 将输入数据传入嵌入层，得到词向量表示
        embedding = self.embedding(input_ids)

        outputs = [cnn(embedding.permute(0,2,1)) for cnn in self.cnns]
        output = torch.concat(outputs, dim=1)
        output = output.permute(0,2,1)

        # 使用 einsum 来生成 word-pair representations
        # 'bik,bjk->bijK' 这个等式意味着对于每个批次中的每对序列位置，我们复制相应的hidden state
        word_pair_feature = torch.einsum('bik,bjk->bijk', [output, output])
        triple_logits = self.triple_linear(word_pair_feature)
        scores = self.score_linear(word_pair_feature).squeeze(-1)
        loss = None
        if triple_labels is not None:
            loss = self.loss_compute(triple_logits, scores,
                                     triple_labels,
                                     score_labels
                                     )
        output = ModelOutput(
            loss=loss,
            logits=triple_logits,
            scores=scores,
        )
        return output

    def loss_compute(self, triple_logits, scores,
                     triple_labels,
                     score_labels
                     ):
        # 创建上三角掩码
        batch_size = triple_logits.shape[0]
        seq_len = triple_logits.shape[1]
        mask_up = torch.triu(torch.ones(seq_len, seq_len), diagonal=0).bool()

        # 扩展掩码到 batch_size
        mask_up = mask_up.unsqueeze(0).expand(batch_size, -1, -1)
        loss_fn1 = nn.CrossEntropyLoss()

        # 使用掩码过滤 logits 和 labels
        num_classes = triple_logits.shape[-1]

        mask_triple_logits = triple_logits[mask_up.unsqueeze(-1).expand_as(triple_logits)].view(-1, num_classes)
        mask_triple_labels = triple_labels[mask_up].view(-1)
        triple_loss = loss_fn1(mask_triple_logits, mask_triple_labels)

        # 分数loss
        loss_fn2 = nn.MSELoss()
        score_loss = loss_fn2(scores, score_labels)

        loss = triple_loss + score_loss
        return loss



if __name__ == '__main__':
    model = GTSBiLSTM(num_labels=16)
    ids, mask = torch.randint(0, 100, (2, 10)), torch.randint(0, 1, (2, 10))
    output = model(ids, )
    print(output)

