import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertPreTrainedModel, BertConfig
from transformers.modeling_outputs import ModelOutput
import torch.nn.functional as F


class GTSBert(BertPreTrainedModel):
    def __init__(self, config, ):
        super(GTSBert, self).__init__(config)
        self.config = config
        self.bert = BertModel(config=config)
        self.triple_linear = nn.Linear(config.hidden_size, config.num_labels)

        self.score_linear = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask,
                triple_labels=None,
                score_labels=None,

                ):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, )

        bert_feature = bert_outputs.last_hidden_state
        # 使用 einsum 来生成 word-pair representations
        # 'bik,bjk->bijK' 这个等式意味着对于每个批次中的每对序列位置，我们复制相应的hidden state
        word_pair_feature = torch.einsum('bik,bjk->bijk', [bert_feature, bert_feature])
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


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class RefiningStrategy(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_ratio=0.1):
        super(RefiningStrategy, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.in_dim * 5, self.out_dim)
        # self.W = nn.Linear(self.hidden_dim * 2 + self.edge_dim * 1, self.dim_e)

    def forward(self, edge, node1, node2):
        batch, seq, seq, edge_dim = edge.shape
        node = torch.cat([node1, node2], dim=-1)

        edge_diag = torch.diagonal(edge, offset=0, dim1=1, dim2=2).permute(0, 2, 1).contiguous()
        edge_i = edge_diag.unsqueeze(1).expand(batch, seq, seq, edge_dim)
        edge_j = edge_i.permute(0, 2, 1, 3).contiguous()
        edge = self.W(torch.cat([edge, edge_i, edge_j, node], dim=-1))

        # edge = self.W(torch.cat([edge, node], dim=-1))

        return edge


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, in_dim, out_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pooling = pooling
        self.layernorm = LayerNorm(self.in_dim)
        self.W = nn.Linear(self.in_dim, self.out_dim)
        self.highway = RefiningStrategy(in_dim, out_dim, dropout_ratio=0.1)

    def forward(self, edge_feature, gcn_inputs, self_loop):
        batch, seq, dim = gcn_inputs.shape
        edge_feature = edge_feature.permute(0, 3, 1, 2)

        gcn_inputs = gcn_inputs.unsqueeze(1).expand(batch, self.in_dim, seq, dim)

        edge_feature = edge_feature + self_loop
        Ax = torch.matmul(edge_feature, gcn_inputs)
        if self.pooling == 'avg':
            Ax = Ax.mean(dim=1)
        elif self.pooling == 'max':
            Ax, _ = Ax.max(dim=1)
        elif self.pooling == 'sum':
            Ax = Ax.sum(dim=1)
        # Ax: [batch, seq, dim]
        gcn_outputs = self.W(Ax)
        gcn_outputs = self.layernorm(gcn_outputs)
        weights_gcn_outputs = F.relu(gcn_outputs)

        node_outputs = weights_gcn_outputs
        edge_feature = edge_feature.permute(0, 2, 3, 1).contiguous()
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
        edge_outputs = self.highway(edge_feature, node_outputs1, node_outputs2)

        return node_outputs, edge_outputs


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class EMCGCN(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super(EMCGCN, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.dropout_output = torch.nn.Dropout(config.hidden_dropout_prob)

        self.pos_emb = torch.nn.Embedding(677, config.hidden_size // 3, padding_idx=0)
        self.dep_emb = torch.nn.Embedding(16, config.hidden_size // 3, padding_idx=0)
        self.dis_emb = torch.nn.Embedding(128, config.hidden_size // 3, padding_idx=0)

        self.ap_fc = nn.Linear(config.hidden_size, config.hidden_size // 3)
        self.op_fc = nn.Linear(config.hidden_size, config.hidden_size // 3)
        self.triplet_biaffine = Biaffine(config.hidden_size // 3, config.hidden_size // 3, )

        self.edge_dense = nn.Linear(config.hidden_size // 3 * 4, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.num_layers = 2
        self.gcn_layers = nn.ModuleList()

        self.layernorm = LayerNorm(config.hidden_size)

        for i in range(self.num_layers):
            self.gcn_layers.append(
                GraphConvLayer(config.hidden_size, config.hidden_size))

        self.triple_linear = nn.Linear(config.hidden_size, config.num_labels)

        self.score_linear = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, pos, dep, dis, triple_labels=None, score_labels=None):
        batch, seq = attention_mask.shape

        output = self.bert(input_ids, attention_mask)
        bert_feature = output.last_hidden_state
        bert_feature = self.dropout_output(bert_feature)

        # * multi-feature
        pos_emb = self.pos_emb(pos)
        dep_emb = self.dep_emb(dep)
        dis_emb = self.dis_emb(dis)

        # BiAffine
        # [batch, seq_len, hidden_dim], [batch, seq_len, hidden_dim] -> [batch, seq_len, seq_len, hidden_dim]
        ap_node = F.relu(self.ap_fc(bert_feature))
        op_node = F.relu(self.op_fc(bert_feature))
        biaffine_edge_feature = self.triplet_biaffine(ap_node, op_node)

        edge_feature = torch.cat([biaffine_edge_feature, pos_emb, dep_emb, dis_emb], dim=-1)
        edge_feature = F.relu(self.edge_dense(edge_feature))
        gcn_input = F.relu(self.dense(bert_feature))
        gcn_outputs = gcn_input

        self_loop = []
        for _ in range(batch):
            self_loop.append(torch.eye(seq))
        self_loop = torch.stack(self_loop).to(gcn_outputs.device).unsqueeze(1).expand(batch, self.config.hidden_size,
                                                                                      seq, seq)

        for _layer in range(self.num_layers):
            gcn_outputs, edge_feature = self.gcn_layers[_layer](edge_feature, gcn_outputs,
                                                                self_loop)  # [batch, seq, dim]

        triple_logits = self.triple_linear(edge_feature)
        scores = self.score_linear(edge_feature).squeeze(-1)
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


class AttentionBasedFeatureFusion(nn.Module):
    def __init__(self, layer_count, hidden_dim=128):
        super(AttentionBasedFeatureFusion, self).__init__()
        self.layer_count = layer_count
        self.hidden_dim = hidden_dim
        self.query = nn.LazyLinear(hidden_dim)
        self.key = nn.LazyLinear(hidden_dim)
        self.value = nn.LazyLinear(hidden_dim)
        self.weight_network = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.LazyLinear(layer_count)
        )

    def forward(self, layer_outputs):
        # 检查输入的层数是否与初始化时期望的一致
        assert len(layer_outputs) == self.layer_count, "输入层的数量与期望的层数不匹配。"
        self.feature_dim = layer_outputs[0].shape[-1]
        # 使用query、key和value转换每一层的平均输出
        # avg_outputs = torch.mean(torch.stack(layer_outputs, dim=1), dim=2)  # (batch_size, layer_count, feature_dim)
        avg_outputs = torch.stack([output[:, 0] for output in layer_outputs], dim=1)
        queries = self.query(avg_outputs)  # (batch_size, layer_count, hidden_dim)
        keys = self.key(avg_outputs)  # (batch_size, layer_count, hidden_dim)
        values = self.value(avg_outputs)  # (batch_size, layer_count, hidden_dim)

        # 计算注意力分数
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.feature_dim ** 0.5)  # Scale
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, layer_count, layer_count)

        # 计算加权的值
        weighted_values = torch.matmul(attention_weights, values)  # (batch_size, layer_count, hidden_dim)

        # 使用权重网络计算每层的最终权重
        final_weights = self.weight_network(weighted_values.reshape(-1, self.layer_count * self.hidden_dim))
        final_weights = F.softmax(final_weights.reshape(-1, self.layer_count, 1, 1),
                                  dim=-1)  # (batch_size, layer_count, 1, 1)

        # 将层输出堆叠成一个新的维度，形状变为(batch_size, layer_count, seq_len, feature_dim)
        stacked_outputs = torch.stack(layer_outputs, dim=1)

        # 使用广播机制进行加权融合
        fused_feature = torch.sum(final_weights * stacked_outputs, dim=1)

        return fused_feature


class EMCGCN_ABFF(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super(EMCGCN_ABFF, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.hidden_layers = 6
        self.abff = AttentionBasedFeatureFusion(layer_count=self.hidden_layers, hidden_dim=config.hidden_size)
        self.dropout_output = torch.nn.Dropout(config.hidden_dropout_prob)

        self.pos_emb = torch.nn.Embedding(677, config.hidden_size // 3, padding_idx=0)
        self.dep_emb = torch.nn.Embedding(16, config.hidden_size // 3, padding_idx=0)
        self.dis_emb = torch.nn.Embedding(128, config.hidden_size // 3, padding_idx=0)

        self.ap_fc = nn.Linear(config.hidden_size, config.hidden_size // 3)
        self.op_fc = nn.Linear(config.hidden_size, config.hidden_size // 3)
        self.triplet_biaffine = Biaffine(config.hidden_size // 3, config.hidden_size // 3, )

        self.edge_dense = nn.Linear(config.hidden_size // 3 * 4, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.num_layers = 2
        self.gcn_layers = nn.ModuleList()

        self.layernorm = LayerNorm(config.hidden_size)


        for i in range(self.num_layers):
            self.gcn_layers.append(
                GraphConvLayer(config.hidden_size, config.hidden_size))

        self.triple_linear = nn.Linear(config.hidden_size, config.num_labels)

        self.score_linear = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, pos, dep, dis, triple_labels=None, score_labels=None):
        batch, seq = attention_mask.shape

        output = self.bert(input_ids, attention_mask, output_hidden_states=True)
        hidden_states = output.hidden_states[-self.hidden_layers:]
        bert_feature = self.abff(hidden_states)
        bert_feature = self.dropout_output(bert_feature)

        # * multi-feature
        pos_emb = self.pos_emb(pos)
        dep_emb = self.dep_emb(dep)
        dis_emb = self.dis_emb(dis)

        # BiAffine
        ap_node = F.relu(self.ap_fc(bert_feature))
        op_node = F.relu(self.op_fc(bert_feature))
        biaffine_edge_feature = self.triplet_biaffine(ap_node, op_node)

        edge_feature = torch.cat([biaffine_edge_feature, pos_emb, dep_emb, dis_emb], dim=-1)
        edge_feature = F.relu(self.edge_dense(edge_feature))
        gcn_input = F.relu(self.dense(bert_feature))
        gcn_outputs = gcn_input

        self_loop = []
        for _ in range(batch):
            self_loop.append(torch.eye(seq))
        self_loop = torch.stack(self_loop).to(gcn_outputs.device).unsqueeze(1).expand(batch, self.config.hidden_size,
                                                                                      seq, seq)

        for _layer in range(self.num_layers):
            gcn_outputs, edge_feature = self.gcn_layers[_layer](edge_feature, gcn_outputs,
                                                                self_loop)  # [batch, seq, dim]

        triple_logits = self.triple_linear(edge_feature)
        scores = self.score_linear(edge_feature).squeeze(-1)
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

#
# class SAM(nn.Module):
#     def __init__(self, kernel_size=3, bias=False):
#         super(SAM, self).__init__()
#         self.bias = bias
#         self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding='same',
#                               dilation=1,
#                               bias=self.bias)
#
#     def forward(self, x):
#         max = torch.max(x, 1)[0].unsqueeze(1)
#         avg = torch.mean(x, 1).unsqueeze(1)
#         concat = torch.cat((max, avg), dim=1)
#         output = self.conv(concat)
#         output = F.sigmoid(output) * x
#         return output
#
#
# class CAM(nn.Module):
#     def __init__(self, channels, r):
#         super(CAM, self).__init__()
#         self.channels = channels
#         self.r = r
#         self.linear = nn.Sequential(
#             nn.Linear(in_features=self.channels, out_features=self.channels // self.r, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=self.channels // self.r, out_features=self.channels, bias=True))
#
#     def forward(self, x):
#         max = F.adaptive_max_pool2d(x, output_size=1)
#         avg = F.adaptive_avg_pool2d(x, output_size=1)
#         b, c, _, _ = x.size()
#         linear_max = self.linear(max.view(b, c)).view(b, c, 1, 1)
#         linear_avg = self.linear(avg.view(b, c)).view(b, c, 1, 1)
#         output = linear_max + linear_avg
#         output = F.sigmoid(output) * x
#         return output
#
#
# class CSA(nn.Module):
#     def __init__(self, channels, r, kernel_size=3):
#         super(CSA, self).__init__()
#         self.channels = channels
#         self.r = r
#         self.sam = SAM(kernel_size=kernel_size, bias=False)
#         self.cam = CAM(channels=self.channels, r=self.r)
#
#     def forward(self, x):
#         output = self.cam(x)
#         output = self.sam(output)
#         return output + x
#
#
# class ConvCSA(nn.Module):
#     def __init__(self, in_dim, ratio=4, kernel_sizes=[2, 3, 4, 5]):
#         super(ConvCSA, self).__init__()
#         blocks = []
#         n = len(kernel_sizes)
#         for k in kernel_sizes:
#             blocks.append(
#                 nn.Sequential(
#                     # nn.Linear(in_dim, in_dim//n),
#                     nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=k, padding='same'),
#                     nn.BatchNorm2d(in_dim),
#                     CSA(in_dim, r=ratio, kernel_size=k)
#                 )
#             )
#         self.linear = nn.Linear(in_dim * n, in_dim)
#         self.blocks = nn.ModuleList(blocks)
#         # self.conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding='same')
#         # self.csa = CSA(in_dim, r=ratio, kernel_size=3)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1).unsqueeze(-1)
#         # x = self.conv(x)
#         # out = self.csa(x)
#         outputs = []
#         for block in self.blocks:
#             output = block(x)
#             outputs.append(output)
#         out = torch.cat(outputs, dim=1)
#         out = out.squeeze(-1).permute(0, 2, 1)
#         out = self.linear(out)
#         out = self.relu(out)
#         return out

#
# class EMCGCN_ABFF_Conv(BertPreTrainedModel):
#     def __init__(self, config: BertConfig):
#         super(EMCGCN_ABFF_Conv, self).__init__(config)
#         self.config = config
#         self.bert = BertModel(config)
#         self.hidden_layers = 6
#         self.abff = AttentionBasedFeatureFusion(layer_count=self.hidden_layers, hidden_dim=config.hidden_size)
#         self.dropout_output = torch.nn.Dropout(config.hidden_dropout_prob)
#         # self.conv_csa = ConvCSA(in_dim=config.hidden_size, ratio=4, kernel_sizes=[2,3,4])
#         self.pos_emb = torch.nn.Embedding(677, config.hidden_size // 3, padding_idx=0)
#         self.dep_emb = torch.nn.Embedding(16, config.hidden_size // 3, padding_idx=0)
#         self.dis_emb = torch.nn.Embedding(128, config.hidden_size // 3, padding_idx=0)
#
#         self.ap_fc = nn.Linear(config.hidden_size, config.hidden_size // 3)
#         self.op_fc = nn.Linear(config.hidden_size, config.hidden_size // 3)
#         self.triplet_biaffine = Biaffine(config.hidden_size // 3, config.hidden_size // 3, )
#
#         self.edge_dense = nn.Linear(config.hidden_size // 3 * 4, config.hidden_size)
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.num_layers = 2
#         self.gcn_layers = nn.ModuleList()
#
#         self.layernorm = LayerNorm(config.hidden_size)
#
#         for i in range(self.num_layers):
#             self.gcn_layers.append(
#                 GraphConvLayer(config.hidden_size, config.hidden_size))
#
#         self.conv1 = nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=3, padding='same')
#         self.csa = CSA(config.hidden_size, r=4, kernel_size=3)
#         self.triple_linear = nn.Linear(2 * config.hidden_size, config.num_labels)
#
#         self.score_linear = nn.Linear(2 * config.hidden_size, 1)
#
#     def forward(self, input_ids, attention_mask, pos, dep, dis, triple_labels=None, score_labels=None):
#         batch, seq = attention_mask.shape
#
#         output = self.bert(input_ids, attention_mask, output_hidden_states=True)
#         hidden_states = output.hidden_states[-self.hidden_layers:]
#         bert_feature = self.abff(hidden_states)
#         bert_feature = self.dropout_output(bert_feature)
#         # bert_feature = self.conv_csa(bert_feature)
#         # * multi-feature
#         pos_emb = self.pos_emb(pos)
#         dep_emb = self.dep_emb(dep)
#         dis_emb = self.dis_emb(dis)
#
#         # BiAffine
#         ap_node = F.relu(self.ap_fc(bert_feature))
#         op_node = F.relu(self.op_fc(bert_feature))
#         biaffine_edge_feature = self.triplet_biaffine(ap_node, op_node)
#
#         edge_feature = torch.cat([biaffine_edge_feature, pos_emb, dep_emb, dis_emb], dim=-1)
#         edge_feature = F.relu(self.edge_dense(edge_feature))
#         gcn_input = F.relu(self.dense(bert_feature))
#         gcn_outputs = gcn_input
#
#         self_loop = []
#         for _ in range(batch):
#             self_loop.append(torch.eye(seq))
#         self_loop = torch.stack(self_loop).to(gcn_outputs.device).unsqueeze(1).expand(batch, self.config.hidden_size,
#                                                                                       seq, seq)
#         conv_feature = self.conv1(edge_feature.permute(0, 3, 1, 2))
#         # csa_feature = self.csa(edge_feature.permute(0,3,1,2))
#
#         for _layer in range(self.num_layers):
#             gcn_outputs, edge_feature = self.gcn_layers[_layer](edge_feature, gcn_outputs,
#                                                                 self_loop)  # [batch, seq, dim]
#         edge_feature = torch.cat([conv_feature.permute(0, 2, 3, 1), edge_feature], dim=-1)
#         triple_logits = self.triple_linear(edge_feature)
#         scores = self.score_linear(edge_feature).squeeze(-1)
#         loss = None
#         if triple_labels is not None:
#             loss = self.loss_compute(triple_logits, scores,
#                                      triple_labels,
#                                      score_labels
#                                      )
#         output = ModelOutput(
#             loss=loss,
#             logits=triple_logits,
#             scores=scores,
#         )
#         return output
#
#     def loss_compute(self, triple_logits, scores,
#                      triple_labels,
#                      score_labels
#                      ):
#         # 创建上三角掩码
#         batch_size = triple_logits.shape[0]
#         seq_len = triple_logits.shape[1]
#         mask_up = torch.triu(torch.ones(seq_len, seq_len), diagonal=0).bool()
#
#         # 扩展掩码到 batch_size
#         mask_up = mask_up.unsqueeze(0).expand(batch_size, -1, -1)
#         loss_fn1 = nn.CrossEntropyLoss()
#
#         # 使用掩码过滤 logits 和 labels
#         num_classes = triple_logits.shape[-1]
#
#         mask_triple_logits = triple_logits[mask_up.unsqueeze(-1).expand_as(triple_logits)].view(-1, num_classes)
#         mask_triple_labels = triple_labels[mask_up].view(-1)
#         triple_loss = loss_fn1(mask_triple_logits, mask_triple_labels)
#
#         # 分数loss
#         loss_fn2 = nn.MSELoss()
#         score_loss = loss_fn2(scores, score_labels)
#
#         loss = triple_loss + score_loss
#         return loss


class EMCGCN_ABFF_Word(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super(EMCGCN_ABFF_Word, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.hidden_layers = 6
        self.abff = AttentionBasedFeatureFusion(layer_count=self.hidden_layers, hidden_dim=config.hidden_size)
        self.dropout_output = torch.nn.Dropout(config.hidden_dropout_prob)

        self.word_embedding = nn.Embedding(5172, config.hidden_size)
        self.lstm = nn.GRU(config.hidden_size, config.hidden_size//2, bidirectional=True, num_layers=2)
        self.attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=8)
        self.triplet_biaffine_word = Biaffine(config.hidden_size // 3, config.hidden_size // 3, )
        self.ap_fc_word = nn.Linear(config.hidden_size, config.hidden_size // 3)
        self.op_fc_word = nn.Linear(config.hidden_size, config.hidden_size // 3)

        self.pos_emb = torch.nn.Embedding(677, config.hidden_size // 3, padding_idx=0)
        self.dep_emb = torch.nn.Embedding(16, config.hidden_size // 3, padding_idx=0)
        self.dis_emb = torch.nn.Embedding(128, config.hidden_size // 3, padding_idx=0)

        self.ap_fc = nn.Linear(config.hidden_size, config.hidden_size // 3)
        self.op_fc = nn.Linear(config.hidden_size, config.hidden_size // 3)
        self.triplet_biaffine = Biaffine(config.hidden_size // 3, config.hidden_size // 3, )
        self.channel_attention = nn.MultiheadAttention(config.hidden_size // 3 * 5, num_heads=1)
        self.edge_dense = nn.Linear(config.hidden_size // 3 * 5, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.num_layers = 2
        self.gcn_layers = nn.ModuleList()

        # self.layernorm = LayerNorm(config.hidden_size)
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size//3*5)
        for i in range(self.num_layers):
            self.gcn_layers.append(
                GraphConvLayer(config.hidden_size, config.hidden_size))

        self.triple_linear = nn.Linear(config.hidden_size, config.num_labels)

        self.score_linear = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask, pos, dep, dis, word_ids, triple_labels=None, score_labels=None):
        batch, seq = attention_mask.shape

        output = self.bert(input_ids, attention_mask, output_hidden_states=True)
        hidden_states = output.hidden_states[-self.hidden_layers:]
        bert_feature = self.abff(hidden_states)
        bert_feature = self.dropout_output(bert_feature)

        word_feature = self.word_embedding(word_ids)
        # word_feature, _ = self.lstm(word_feature)
        # word_feature = self.layernorm1(word_feature)
        # word_feature, _ = self.attention(word_feature, word_feature, word_feature)
        # word_feature = self.dropout_output(word_feature)
        ap_node_word = F.relu(self.ap_fc_word(word_feature))
        op_node_word = F.relu(self.op_fc_word(word_feature))
        biaffine_word_feature = self.triplet_biaffine_word(ap_node_word, op_node_word)

        # bert_feature = word_feature + bert_feature
        # * multi-feature
        pos_emb = self.pos_emb(pos)
        dep_emb = self.dep_emb(dep)
        dis_emb = self.dis_emb(dis)

        # BiAffine
        ap_node = F.relu(self.ap_fc(bert_feature))
        op_node = F.relu(self.op_fc(bert_feature))
        biaffine_edge_feature = self.triplet_biaffine(ap_node, op_node)

        edge_feature = torch.cat([biaffine_edge_feature, biaffine_word_feature, pos_emb, dep_emb, dis_emb], dim=-1)

        # b, l, l, h = edge_feature.shape
        # flatten_edge_feature = edge_feature.view(b, -1, h)
        # flatten_edge_feature = self.layernorm2(flatten_edge_feature)
        # edge_feature, _ = self.channel_attention(flatten_edge_feature, flatten_edge_feature, flatten_edge_feature)
        # edge_feature = edge_feature.view(b, l, l, h)


        edge_feature = F.relu(self.edge_dense(edge_feature))
        gcn_input = F.relu(self.dense(bert_feature))
        gcn_outputs = gcn_input

        self_loop = []
        for _ in range(batch):
            self_loop.append(torch.eye(seq))
        self_loop = torch.stack(self_loop).to(gcn_outputs.device).unsqueeze(1).expand(batch, self.config.hidden_size,
                                                                                      seq, seq)

        for _layer in range(self.num_layers):
            gcn_outputs, edge_feature = self.gcn_layers[_layer](edge_feature, gcn_outputs,
                                                                self_loop)  # [batch, seq, dim]

        triple_logits = self.triple_linear(edge_feature)
        scores = self.score_linear(edge_feature).squeeze(-1)
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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EMCGCN_ABFF_Word_ATT(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super(EMCGCN_ABFF_Word_ATT, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.hidden_layers = 6
        self.abff = AttentionBasedFeatureFusion(layer_count=self.hidden_layers, hidden_dim=config.hidden_size)
        self.dropout_output = torch.nn.Dropout(config.hidden_dropout_prob)

        self.word_embedding = nn.Embedding(5172, config.hidden_size)
        # self.lstm = nn.GRU(config.hidden_size, config.hidden_size//2, bidirectional=True, num_layers=2)
        # self.attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=8)
        self.triplet_biaffine_word = Biaffine(config.hidden_size // 3, config.hidden_size // 3, )
        self.ap_fc_word = nn.Linear(config.hidden_size, config.hidden_size // 3)
        self.op_fc_word = nn.Linear(config.hidden_size, config.hidden_size // 3)

        self.pos_emb = torch.nn.Embedding(677, config.hidden_size // 3, padding_idx=0)
        self.dep_emb = torch.nn.Embedding(16, config.hidden_size // 3, padding_idx=0)
        self.dis_emb = torch.nn.Embedding(128, config.hidden_size // 3, padding_idx=0)

        self.ap_fc = nn.Linear(config.hidden_size, config.hidden_size // 3)
        self.op_fc = nn.Linear(config.hidden_size, config.hidden_size // 3)
        self.triplet_biaffine = Biaffine(config.hidden_size // 3, config.hidden_size // 3, )
        # self.channel_attention = nn.MultiheadAttention(config.hidden_size // 3 * 5, num_heads=1)
        self.se = SELayer(channel=config.hidden_size // 3 * 5, reduction=4)
        self.edge_dense = nn.Linear(config.hidden_size // 3 * 5, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.num_layers = 2
        self.gcn_layers = nn.ModuleList()

        # self.layernorm = LayerNorm(config.hidden_size)
        # self.layernorm1 = nn.LayerNorm(config.hidden_size)
        # self.layernorm2 = nn.LayerNorm(config.hidden_size//3*5)
        for i in range(self.num_layers):
            self.gcn_layers.append(
                GraphConvLayer(config.hidden_size, config.hidden_size))

        self.triple_linear = nn.Linear(config.hidden_size, config.num_labels)

        self.score_linear = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask, pos, dep, dis, word_ids, triple_labels=None, score_labels=None):
        batch, seq = attention_mask.shape

        output = self.bert(input_ids, attention_mask, output_hidden_states=True)
        hidden_states = output.hidden_states[-self.hidden_layers:]
        bert_feature = self.abff(hidden_states)
        bert_feature = self.dropout_output(bert_feature)

        word_feature = self.word_embedding(word_ids)
        # word_feature, _ = self.lstm(word_feature)
        # word_feature = self.layernorm1(word_feature)
        # word_feature, _ = self.attention(word_feature, word_feature, word_feature)
        # word_feature = self.dropout_output(word_feature)
        ap_node_word = F.relu(self.ap_fc_word(word_feature))
        op_node_word = F.relu(self.op_fc_word(word_feature))
        biaffine_word_feature = self.triplet_biaffine_word(ap_node_word, op_node_word)

        # bert_feature = word_feature + bert_feature
        # * multi-feature
        pos_emb = self.pos_emb(pos)
        dep_emb = self.dep_emb(dep)
        dis_emb = self.dis_emb(dis)

        # BiAffine
        ap_node = F.relu(self.ap_fc(bert_feature))
        op_node = F.relu(self.op_fc(bert_feature))
        biaffine_edge_feature = self.triplet_biaffine(ap_node, op_node)

        edge_feature = torch.cat([biaffine_edge_feature, biaffine_word_feature, pos_emb, dep_emb, dis_emb], dim=-1)

        # b, l, l, h = edge_feature.shape
        # flatten_edge_feature = edge_feature.view(b, -1, h)
        # flatten_edge_feature = self.layernorm2(flatten_edge_feature)
        # edge_feature, _ = self.channel_attention(flatten_edge_feature, flatten_edge_feature, flatten_edge_feature)
        # edge_feature = edge_feature.view(b, l, l, h)

        edge_feature = self.se(edge_feature.permute(0,3,1,2)).permute(0,2,3,1)


        edge_feature = F.relu(self.edge_dense(edge_feature))
        gcn_input = F.relu(self.dense(bert_feature))
        gcn_outputs = gcn_input

        self_loop = []
        for _ in range(batch):
            self_loop.append(torch.eye(seq))
        self_loop = torch.stack(self_loop).to(gcn_outputs.device).unsqueeze(1).expand(batch, self.config.hidden_size,
                                                                                      seq, seq)

        for _layer in range(self.num_layers):
            gcn_outputs, edge_feature = self.gcn_layers[_layer](edge_feature, gcn_outputs,
                                                                self_loop)  # [batch, seq, dim]

        triple_logits = self.triple_linear(edge_feature)
        scores = self.score_linear(edge_feature).squeeze(-1)
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
    # model = GTSBert.from_pretrained('D:/plm/bert-base-chinese')
    # ids, mask = torch.randint(0, 100, (2, 10)), torch.randint(0, 1, (2, 10))
    # model(ids, mask)
    pass
