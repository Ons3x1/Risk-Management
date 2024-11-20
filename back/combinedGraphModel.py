import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch.nn import Linear

class CombinedGraphModel(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads):
        super(CombinedGraphModel, self).__init__()
        self.sage_conv1 = SAGEConv(in_feats, hidden_feats)
        self.gat_conv1 = GATConv(hidden_feats, hidden_feats, heads=num_heads, concat=True)
        self.sage_conv2 = SAGEConv(hidden_feats * num_heads, hidden_feats)
        self.gat_conv2 = GATConv(hidden_feats, out_feats, heads=1, concat=False)
        self.dense = Linear(out_feats, out_feats)  # Dense layer

    def forward(self, data):
        h_sage1 = F.relu(self.sage_conv1(data.x, data.edge_index))
        h_gat1 = F.relu(self.gat_conv1(h_sage1, data.edge_index))
        h_sage2 = F.relu(self.sage_conv2(h_gat1, data.edge_index))
        h_gat2 = self.gat_conv2(h_sage2, data.edge_index)
        h_dense = self.dense(h_gat2)  # Pass through dense layer
        return h_dense, h_sage1, h_gat1, h_sage2