# -----------------------------
# Discriminator (GAT + level-wise pooling)
# -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATConv


class AIGDiscriminator(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, heads=4):
        super().__init__()
        # use edge_dim to include edge_attr (1-dim)
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, edge_dim=1, concat=True)
        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=1, concat=True)
        self.lin = nn.Linear(hidden_dim, hidden_dim//2)
        self.out = nn.Linear(hidden_dim//2, 1)

    def forward(self, data: Data):
        # data: padded Batch or Batch with x, edge_index, edge_attr, node_depth
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float() if (hasattr(data, 'edge_attr') and data.edge_attr is not None) else None
        # GAT requires edge_attr shape [E, edge_dim], ok
        h = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        h = F.leaky_relu(self.conv2(h, edge_index, edge_attr))
        # level-wise pooling using node_depth (node_depth may contain -1 for padded)
        if hasattr(data, 'node_depth') and data.node_depth is not None:
            nd = data.node_depth
            maxd = int(torch.max(nd).item()) if (nd.numel()>0 and torch.max(nd).item()>=0) else 0
            level_vecs = []
            for l in range(0, maxd+1):
                mask = (nd == l)
                if mask.sum() == 0:
                    # pad zero vector
                    level_vecs.append(torch.zeros((2*h.size(1),), device=h.device))
                    continue
                sub = h[mask]
                mean_v = sub.mean(dim=0)
                max_v = sub.max(dim=0)[0]
                level_vecs.append(torch.cat([mean_v, max_v], dim=0))
            # stack and aggregate across levels (a simple mean over levels)
            level_mat = torch.stack(level_vecs, dim=0)  # [L, 2H]
            graph_repr = level_mat.mean(dim=0)  # [2H]
            # reduce dimension
            g = F.relu(self.lin(graph_repr))
            out = self.out(g)
            return out.view(-1)
        else:
            # fallback: global mean pool
            g = h.mean(dim=0)
            g = F.relu(self.lin(g))
            return self.out(g).view(-1)

