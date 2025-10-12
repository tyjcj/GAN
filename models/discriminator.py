# models/discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter
from torch_geometric.data import Data

# We will implement level-wise pooling manually without relying on global_mean_pool/global_max_pool
# to better control the behavior with node_depth. This discriminator uses edge_attr via GATConv(edge_dim=1).


class AIGDiscriminator(nn.Module):
    """
    Discriminator / critic for AIG graphs.
    - Uses GATConv with edge_dim=1 so edge_attr is incorporated in message computation.
    - Implements level-wise pooling based on data.node_depth:
        For each level l, compute mean and max pooling over node embeddings at that level,
        then aggregate level vectors (mean over levels) and feed to MLP to produce scalar critic.
    """

    def __init__(self, in_dim: int = 2, hidden_dim: int = 128, heads: int = 4):
        super().__init__()
        # GATConv supports edge_dim argument (PyG >= some versions)
        # We map in_dim->hidden_dim with attention and use edge_attr of dim=1
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.hidden_dim = hidden_dim
        self.heads = heads

        self.gat1 = GATConv(in_dim, hidden_dim // heads, heads=heads, edge_dim=1, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=1, concat=True)

        # after level-wise pooling the vector size is 2 * hidden_dim (mean+max)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_dim // 2, 1)

    def forward(self, data: Data):
        """
        Expect data.x: [N, in_dim], data.edge_index: [2, E], data.edge_attr: [E, 1], data.node_depth: [N]
        If node_depth not present, fallback to global mean+max pooling.
        """
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float() if (hasattr(data, "edge_attr") and data.edge_attr is not None) else None

        # If this is a Batch, node indices already concatenated; node_depth should be present and relative to batch,
        # but our train pipeline uses padded single graphs or batches of padded graphs.
        # We'll treat data.node_depth as a 1D tensor aligned with x.
        # Run GAT convs
        if edge_attr is None:
            # If for some reason edge_attr missing, create zeros
            E = edge_index.size(1) if (edge_index is not None and edge_index.numel() > 0) else 0
            edge_attr = torch.zeros((E, 1), dtype=x.dtype, device=x.device)

        h = F.elu(self.gat1(x, edge_index, edge_attr))
        h = F.elu(self.gat2(h, edge_index, edge_attr))

        # Level-wise pooling
        if hasattr(data, "node_depth") and data.node_depth is not None:
            nd = data.node_depth
            # nd might have -1 for padded nodes; we ignore -1 levels
            valid_mask = (nd >= 0)
            if valid_mask.sum() == 0:
                # fallback to global pooling
                mean_v = h.mean(dim=0, keepdim=True)
                max_v = h.max(dim=0)[0].unsqueeze(0)
                graph_repr = torch.cat([mean_v, max_v], dim=1)  # [1, 2H]
            else:
                nd_valid = nd.clone()
                nd_valid[~valid_mask] = -1
                max_level = int(nd_valid.max().item())
                level_vecs = []
                for l in range(0, max_level + 1):
                    mask = (nd_valid == l)
                    if mask.sum() == 0:
                        # append zero vector to keep position; this preserves level index semantics
                        level_vecs.append(torch.zeros((2 * self.hidden_dim,), device=h.device))
                        continue
                    sub = h[mask]
                    mean_v = sub.mean(dim=0)
                    max_v = sub.max(dim=0)[0]
                    level_vecs.append(torch.cat([mean_v, max_v], dim=0))
                # stack levels -> [L, 2H]
                level_mat = torch.stack(level_vecs, dim=0)
                # aggregate across levels (mean)
                graph_repr = level_mat.mean(dim=0, keepdim=True)  # [1, 2H]
        else:
            # fallback: global mean+max pooling
            mean_v = h.mean(dim=0, keepdim=True)
            max_v = h.max(dim=0)[0].unsqueeze(0)
            graph_repr = torch.cat([mean_v, max_v], dim=1)  # [1, 2H]

        g = self.mlp(graph_repr)  # [1, hidden/2]
        out = self.out(g)  # [1,1]
        return out.view(-1)  # scalar per graph
