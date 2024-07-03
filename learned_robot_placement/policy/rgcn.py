import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv
from dgl.nn.pytorch.conv.gatconv import GATConv
from dgl.nn.pytorch.conv.graphconv import GraphConv
from dgl.nn.pytorch.conv.gatconv import GATConv
from learned_robot_placement.policy.helpers import mlp
from learned_robot_placement.policy.he_models.transformer import Transformer
import dgl
from learned_robot_placement.policy.gnn_models import HoRelGAT
import dgl.nn.pytorch as dglnn
import torch.nn.init as init

class RGATLayer(nn.Module):

    def __init__(self,
                 in_feat,
                 out_feat,
                 num_heads,
                 rel_names,
                 activation=None,
                 dropout=0.0,
                 bias=True,):
        super(RGATLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_heads= num_heads
        self.conv = dgl.nn.HeteroGraphConv({
            rel: dgl.nn.pytorch.GATConv(in_feat, out_feat, num_heads=num_heads, bias=bias, allow_zero_in_degree=True)
            for rel in rel_names
        })

    def forward(self, g, h_dict):
        h_dict = self.conv(g, h_dict)
        out_put = {}
        for n_type, h in h_dict.items():
            out_put[n_type] = h.squeeze()
        return out_put


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'


class GcnNet(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, input_dim=1433):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 16)
        self.gcn2 = GraphConvolution(16, 7)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits


class GCN(nn.Module):
    def __init__(self, g, gnn_model, gnn_layers, in_dim, out_dim, hidden_dimensions, num_rels, activation,  final_activation,
                 feat_drop, num_bases=-1):
        super(GCN, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.encoder_dim = [out_dim]
        self.hidden_dimensions = [32]
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.feat_drop = feat_drop
        self.num_bases = num_bases
        self.activation = torch.nn.ReLU()
        self.final_activation = torch.nn.ReLU()
        self.gnn_layers = gnn_layers
        self.gnn_model = gnn_model
        # create RGCN layers
        self.build_model()


    def set_g(self, g):
        self.g = g

    def build_model(self):
        self.layers = nn.ModuleList()
        # 对应一个编码器
        self.encoder = mlp(self.in_dim, self.encoder_dim, last_relu=True)
        # print("self.in_dim=%s, self.encoder_dim=%s"%(self.in_dim, self.encoder_dim))
        i2o = self.build_i2o_layer()
        self.layers.append(i2o)

    def build_i2o_layer(self):
        return dglnn.GraphConv(self.encoder_dim[-1], self.out_dim, allow_zero_in_degree=True)


    def forward(self, state_graph, node_features, edgetypes):
        h = node_features
        h0 = self.encoder(h)
        norm = state_graph.edata['norm']
        output = h0
        for layer in self.layers:
            state_graph = dgl.add_self_loop(state_graph)
            h1 = layer(state_graph, output)
            h1 = h1.reshape(-1, self.out_dim)
            output = output + h1
        ## skip connection???
        return output


class RGCN(nn.Module):
    def __init__(self, g, gnn_model, gnn_layers, in_dim, out_dim, hidden_dimensions, num_rels, activation,  final_activation,
                 feat_drop, num_bases=-1):
        super(RGCN, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.encoder_dim = [out_dim]
        self.hidden_dimensions = [32]
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.feat_drop = feat_drop
        self.num_bases = num_bases
        self.activation = torch.nn.ReLU()
        self.final_activation = torch.nn.ReLU()
        self.gnn_layers = gnn_layers
        self.gnn_model = gnn_model
        if self.gnn_model == 'rgcn':
            self.use_rgcn = True
        elif self.gnn_model == 'gat':
            self.use_gat = True
        elif self.gnn_model == 'gcn':
            self.use_gcn = True
        elif self.gnn_model == 'rgat':
            self.use_rgat = True
        elif self.gnn_model == 'transfomer':
            self.use_transformer = True
        # create RGCN layers
        self.build_model()


    def set_g(self, g):
        self.g = g

    def build_model(self):
        self.layers = nn.ModuleList()
        # 对应一个编码器
        self.encoder = mlp(self.in_dim, self.encoder_dim, last_relu=True)
        i2o = self.build_i2o_layer()
        self.layers.append(i2o)

    def build_input_layer(self):
        print('Building an INPUT  layer of {}x{}'.format(self.in_dim, self.hidden_dimensions[0]))
        return RelGraphConv(self.in_dim, self.hidden_dimensions[0], self.num_rels,
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=F.leaky_relu)


    def build_hidden_layer(self, i):
        print('Building an HIDDEN  layer of {}x{}'.format(self.hidden_dimensions[i], self.hidden_dimensions[i+1]))
        return RelGraphConv(self.hidden_dimensions[i], self.hidden_dimensions[i+1],  self.num_rels,
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=F.leaky_relu)

    def build_output_layer(self):
        print('Building an OUTPUT  layer of {}x{}'.format(self.hidden_dimensions[-1], self.out_dim))
        return RelGraphConv(self.hidden_dimensions[-1], self.out_dim, self.num_rels,
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=self.final_activation)

    def build_i2o_layer(self):
        if self.gnn_model == 'rgcn':
            print('Building an RGCN I2O layer of {}x{}'.format(self.encoder_dim[-1], self.out_dim))
            return RelGraphConv(self.encoder_dim[-1], self.out_dim, self.num_rels,
                                dropout=self.feat_drop, num_bases=self.num_bases, activation=self.final_activation)
        elif  self.gnn_model == 'gcn':
            print('Building an GCN I2O layer of {}x{}'.format(self.encoder_dim[-1], self.out_dim))
            return GraphConv(self.encoder_dim[-1], self.out_dim, activation=self.final_activation)
        elif  self.gnn_model == 'gat':
            print('Building an  GAT I2O  layer of {}x{}'.format(self.encoder_dim[-1], self.out_dim))
            return GATConv(self.encoder_dim[-1], self.out_dim, num_heads=1, activation=self.final_activation)
        elif  self.gnn_model == 'transformer':
            print('Building an  Transformer I2O  layer of {}x{}'.format(self.encoder_dim[-1], self.out_dim))
            return Transformer(self.encoder_dim[-1], self.out_dim, num_heads=1, activation=self.final_activation)

        elif  self.gnn_model == 'rgat':
            print('Building an RGAT I2O  layer of {}x{}'.format(self.encoder_dim[-1], self.out_dim))
            return HoRelGAT(self.encoder_dim[-1], self.out_dim, num_heads=1, num_rels=self.num_rels,
                                dropout=self.feat_drop, num_bases=self.num_bases, activation=self.final_activation)


    def forward(self, state_graph, node_features, edgetypes):
        h = node_features
        h0 = self.encoder(h)
        # norm = state_graph.edata['norm']
        output = h0
        for layer in self.layers:
            if self.gnn_model == 'rgcn':
                h1 = layer(state_graph, output, edgetypes)
                output = output + h1
            elif self.gnn_model == 'gat' or self.gnn_model == 'gcn' or self.gnn_model == 'transformer':
                state_graph = dgl.add_self_loop(state_graph)
                h1 = layer(state_graph, output)
                h1 = h1.reshape(-1, self.out_dim)
                output = output + h1
            elif self.gnn_model == 'rgat':
                # state_graph = dgl.add_self_loop(state_graph)
                h1 = layer(state_graph, output, edgetypes)
                output = output + h1
        ## skip connection???
        return output



