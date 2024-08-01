import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertPreTrainedModel, BertConfig
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
    def __init__(self, in_dim, out_dim,  dropout_ratio=0.1):
        super(RefiningStrategy, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.in_dim*5, self.out_dim)
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

        edge_feature += self_loop
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

        self.pos_emb = torch.nn.Embedding(677, config.hidden_size//3, padding_idx=0)
        self.dep_emb = torch.nn.Embedding(16, config.hidden_size//3, padding_idx=0)
        self.dis_emb = torch.nn.Embedding(128, config.hidden_size//3, padding_idx=0)

        self.triplet_biaffine = Biaffine(config.hidden_size, config.hidden_size,)
        self.ap_fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.op_fc = nn.Linear(config.hidden_size, config.hidden_size)

        self.edge_dense = nn.Linear(config.hidden_size*2, config.hidden_size)
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
        self_loop = torch.stack(self_loop).to(gcn_outputs.device).unsqueeze(1).expand(batch, self.config.hidden_size, seq, seq)

        for _layer in range(self.num_layers):
            gcn_outputs, edge_feature = self.gcn_layers[_layer](edge_feature, gcn_outputs, self_loop)  # [batch, seq, dim]

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


if __name__ == '__main__':
    # Create synthetic data for the test
    batch_size = 2
    seq_len= 4
    hidden_dim = 768

    device = 'cpu'  # or 'cuda' for GPU

    # Initialize the model
    gcn = GraphConvLayer(in_dim=hidden_dim, out_dim=hidden_dim)

    # Synthetic input data
    edge_feature = torch.rand(batch_size, seq_len, seq_len, hidden_dim, device=device)
    gcn_inputs = torch.rand(batch_size, seq_len, hidden_dim, device=device)
    self_loop = torch.eye(seq_len, seq_len, device=device).unsqueeze(0).unsqueeze(1).expand(batch_size, hidden_dim, -1, -1)

    # Forward pass
    node_outputs, edge_outputs = gcn(edge_feature,  gcn_inputs, self_loop)

    # Print the outputs
    print("Node Outputs Shape:", node_outputs.shape)
    print("Edge Outputs Shape:", edge_outputs.shape)

    model = EMCGCN.from_pretrained('D:/plm/bert-base-chinese')
    input_ids = torch.randint(1,20, (2, 4))
    attention_mask = torch.randint(0, 1,(2, 4))
    pos = torch.randint(1, 10, (2, 4,4))
    dep = torch.randint(1, 10, (2, 4,4))
    dis = torch.randint(1, 10, (2, 4,4))

    output = model(input_ids, attention_mask, pos, dep, dis)
    print(output)