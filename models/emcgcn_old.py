import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, BertConfig
from transformers.modeling_outputs import ModelOutput



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
    def __init__(self, hidden_dim, edge_dim, dim_e, dropout_ratio=0.5):
        super(RefiningStrategy, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dim_e = dim_e
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.hidden_dim * 2 + self.edge_dim * 3, self.dim_e)
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

    def __init__(self, gcn_dim, edge_dim, dep_embed_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()
        # self.gcn_dim = gcn_dim
        self.edge_dim = edge_dim
        # self.dep_embed_dim = dep_embed_dim
        self.pooling = pooling
        self.layernorm = LayerNorm(gcn_dim)
        self.W = nn.Linear(gcn_dim, gcn_dim)
        self.gcn_linear = nn.Linear(gcn_dim, edge_dim)
        self.highway = RefiningStrategy(gcn_dim, edge_dim, edge_dim, dropout_ratio=0.5)

    def forward(self, weight_prob_softmax, weight_adj, gcn_inputs, self_loop):
        batch, seq, dim = gcn_inputs.shape
        weight_prob_softmax = weight_prob_softmax.permute(0, 3, 1, 2)
        # gcn_inputs = self.gcn_linear(gcn_inputs)
        gcn_inputs = gcn_inputs.unsqueeze(1).expand(batch, self.edge_dim, seq, dim)

        weight_prob_softmax = weight_prob_softmax + self_loop
        Ax = torch.einsum('bess,besd->besd', weight_prob_softmax, gcn_inputs)
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
        weight_prob_softmax = weight_prob_softmax.permute(0, 2, 3, 1).contiguous()
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
        edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2)
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

class EMCGCN_old(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super(EMCGCN_old, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.dropout_output = torch.nn.Dropout(config.hidden_dropout_prob)

        self.pos_emb = torch.nn.Embedding(677, config.hidden_size // 3, padding_idx=0)
        self.dep_emb = torch.nn.Embedding(16, config.hidden_size // 3, padding_idx=0)
        self.dis_emb = torch.nn.Embedding(128, config.hidden_size // 3, padding_idx=0)

        self.ap_fc = nn.Linear(config.hidden_size, config.hidden_size//3)
        self.op_fc = nn.Linear(config.hidden_size, config.hidden_size//3)
        self.triplet_biaffine = Biaffine(config.hidden_size//3, config.hidden_size//3, )

        self.edge_dense = nn.Linear(config.hidden_size // 3 * 4, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.num_layers = 2
        self.gcn_layers = nn.ModuleList()

        self.layernorm = LayerNorm(config.hidden_size)

        for i in range(self.num_layers):
            self.gcn_layers.append(
                GraphConvLayer(config.hidden_size, config.hidden_size // 3 * 4, config.num_labels))

        self.triple_linear = nn.Linear(config.hidden_size//3 *4, config.num_labels)

        self.score_linear = nn.Linear(config.hidden_size//3*4, 1)

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
        ap_node = F.relu(self.ap_fc(bert_feature))
        op_node = F.relu(self.op_fc(bert_feature))
        biaffine_edge_feature = self.triplet_biaffine(ap_node, op_node)

        edge_feature = torch.cat([biaffine_edge_feature, pos_emb, dep_emb, dis_emb], dim=-1)
        edge_feature = F.relu(self.edge_dense(edge_feature))
        gcn_input = F.relu(self.dense(bert_feature))
        gcn_outputs = gcn_input

        biaffine_edge_feature_softmax = F.softmax(biaffine_edge_feature, dim=-1)
        pos_emb_softmax = F.softmax(pos_emb, dim=-1)
        dep_emb_softmax = F.softmax(dep_emb, dim=-1)
        dis_emb_softmax = F.softmax(dis_emb, dim=-1)

        self_loop = []
        for _ in range(batch):
            self_loop.append(torch.eye(seq))
        self_loop = torch.stack(self_loop).to(gcn_outputs.device).unsqueeze(1).expand(batch, self.config.hidden_size//3 * 4,
                                                                                      seq, seq)
        weight_prob = torch.cat([biaffine_edge_feature, pos_emb, dep_emb, dis_emb,], dim=-1)
        weight_prob_softmax = torch.cat([biaffine_edge_feature_softmax,
                                         pos_emb_softmax,
                                         dep_emb_softmax,
                                         dis_emb_softmax,], dim=-1)

        for _layer in range(self.num_layers):
            gcn_outputs, edge_feature = self.gcn_layers[_layer](weight_prob_softmax, weight_prob, gcn_outputs, self_loop)  # [batch, seq, dim]


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