# GAN/discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional


class LevelWisePool(nn.Module):
    """
    Given node embeddings and node_depth, produce a level-wise sequence tensor:
    output shape: [C, L] where C = 2 * hidden_dim (mean + max pooled per level), L = num_levels
    We'll return as [batch, C, L] for Conv1d processing.
    """
    def __init__(self, hidden_dim: int, max_levels: int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_levels = max_levels

    def forward(self, h: torch.Tensor, node_depth: torch.Tensor, batch_index: Optional[torch.Tensor] = None):
        # h: [N, H]
        # node_depth: [N] (non-negative) ; batch_index: [N] which graph in batch
        if batch_index is None:
            batch_index = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        batch_size = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 1

        # For simplicity we'll process batch items sequentially and stack
        out_list = []
        for b in range(batch_size):
            mask = (batch_index == b)
            h_b = h[mask]
            d_b = node_depth[mask]
            if h_b.size(0) == 0:
                # empty graph, return zeros
                seq = torch.zeros((2 * self.hidden_dim, 1), device=h.device)
                out_list.append(seq)
                continue
            max_d = int(d_b.max().item())
            L = min(max_d + 1, self.max_levels)
            seq_feats = []
            for level in range(L):
                idxs = (d_b == level).nonzero(as_tuple=False).view(-1)
                if idxs.numel() == 0:
                    # zero padding
                    mean_pool = torch.zeros(self.hidden_dim, device=h.device)
                    max_pool = torch.zeros(self.hidden_dim, device=h.device)
                else:
                    nodes = h_b[idxs]
                    mean_pool = nodes.mean(dim=0)
                    max_pool, _ = nodes.max(dim=0)
                seq_feats.append(torch.cat([mean_pool, max_pool], dim=0))  # [2H]
            # seq_feats -> [L, 2H] -> transpose to [2H, L]
            seq = torch.stack(seq_feats, dim=1)
            out_list.append(seq)
        # For batch, pad sequences to same length (max L among graphs)
        max_L = max([s.size(1) for s in out_list])
        padded = []
        for s in out_list:
            if s.size(1) < max_L:
                pad = torch.zeros((s.size(0), max_L - s.size(1)), device=h.device)
                s_p = torch.cat([s, pad], dim=1)
            else:
                s_p = s
            padded.append(s_p.unsqueeze(0))  # [1, C, L]
        return torch.cat(padded, dim=0)  # [B, C, L]


class AIGDiscriminator(nn.Module):
    """
    Discriminator: GCN encoder + level-wise pooling -> Conv1d over levels -> MLP classifier
    """

    def __init__(self, node_in_dim: int = 2, hidden_dim: int = 64, conv_channels: int = 64):
        super().__init__()
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool = LevelWisePool(hidden_dim, max_levels=4096)
        # conv1d: in_channels = 2*hidden_dim, out = conv_channels
        self.conv1d = nn.Conv1d(in_channels=2 * hidden_dim, out_channels=conv_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(conv_channels, conv_channels),
            nn.ReLU(),
            nn.Linear(conv_channels, 1)
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        # batch is a PyG Batch
        x, edge_index = batch.x, batch.edge_index
        node_depth = batch.node_depth
        batch_idx = batch.batch if hasattr(batch, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = F.relu(self.conv1(x.float(), edge_index))
        h = F.relu(self.conv2(h, edge_index))

        # level-wise pooling -> [B, C, L]
        seq = self.pool(h, node_depth, batch_index=batch_idx)  # [B, C, L]
        # conv1d over levels -> output [B, conv_channels, L]
        conv_out = self.act(self.conv1d(seq))
        # global mean pool over levels -> [B, conv_channels]
        pooled = conv_out.mean(dim=2)
        logits = self.fc(pooled).view(-1)  # [B]
        probs = torch.sigmoid(logits)
        return probs
