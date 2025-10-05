import os
import argparse
import random
import glob
import time
import traceback
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch


# -----------------------------
# Generator (conditional)
# -----------------------------
class AIGGenerator(nn.Module):
    """
    Conditional generator:
      - input: template Data (keeps N nodes), noise z (z_dim)
      - for each node v, pick up to 2 predecessors from candidate pool (nodes with lower depth/index)
      - produce edge_attr (0/1) for each chosen edge
    Practical design: compute node embeddings via simple MLP (optionally using template edges).
    Candidate pool size is limited for scalability (candidate_k).
    """
    def __init__(self, node_in_dim=2, hidden_dim=128, z_dim=128, candidate_k=512, depth_perturb=1):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.candidate_k = candidate_k
        self.depth_perturb = depth_perturb

        # simple encoder for template (using node features only for speed)
        self.node_enc = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # combine with z
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, hidden_dim),
            nn.ReLU()
        )
        # scorer for candidate selection (dot product + mlp)
        self.src_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.tgt_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.inv_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2 + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, template: Data, z: torch.Tensor):
        """
        template: Data with x, edge_index (optional), node_depth (optional)
        z: tensor of shape [z_dim] (1D)
        returns: raw Data (not enforced), same N nodes, predicted edges and edge_attr
        """
        device = template.x.device
        x = template.x.float().to(device)
        N = x.size(0)
        # node embeddings
        h = self.node_enc(x)  # [N, H]
        z_expand = z.unsqueeze(0).repeat(N,1)  # [N, z_dim]
        h = self.node_proj(torch.cat([h, z_expand], dim=1))  # [N,H]

        # determine depth info for candidate pools
        if hasattr(template, 'node_depth') and template.node_depth is not None:
            depths = template.node_depth.cpu().tolist()
        else:
            depths = [0 if int(xi[0])==0 else -1 for xi in x.cpu()]

        depth_to_nodes = {}
        maxd = 0
        for i, d in enumerate(depths):
            if d >= 0:
                depth_to_nodes.setdefault(d, []).append(i)
                maxd = max(maxd, d)

        src_list = []
        dst_list = []
        attr_list = []

        for v in range(N):
            t = int(x[v,0].item())
            if t == 0:
                continue  # PI has no predecessors
            # estimate target depth (if known) else use index-based heuristic
            td = depths[v] if (v < len(depths) and depths[v] >= 0) else v//2
            # build candidate pool from depths <= td-1 (allow small perturb)
            pool = []
            for d in range(max(0, td - 3), td + 1):
                pool.extend(depth_to_nodes.get(d, []))
            # ensure exclude self and duplicates
            pool = [u for u in pool if u != v]
            if len(pool) == 0:
                # fallback: allow earlier indices
                pool = [u for u in range(max(0, v-50), v)]
            if len(pool) == 0:
                continue
            # cap candidate size
            if len(pool) > self.candidate_k:
                pool = random.sample(pool, self.candidate_k)

            cand_idx = torch.tensor(pool, dtype=torch.long, device=device)
            cand_emb = h[cand_idx]  # [C, H]
            tgt_emb = h[v:v+1]  # [1, H]
            scores = torch.matmul(self.src_proj(cand_emb), self.tgt_proj(tgt_emb).squeeze(0))  # [C]
            # select up to 2 predecessors
            k = 2 if t == 2 else 1
            k_eff = min(k, scores.size(0))
            topk = torch.topk(scores, k_eff, largest=True)
            selected = [pool[i] for i in topk.indices.tolist()]
            # if less than needed, pad with random choices
            while len(selected) < k:
                selected.append(random.choice(pool))
            # predict inversion per edge
            for u in selected[:k]:
                u_emb = h[u:u+1]
                inv_logit = self.inv_mlp(torch.cat([u_emb.squeeze(0), tgt_emb.squeeze(0), z], dim=0))
                inv_bit = (torch.sigmoid(inv_logit) > 0.5).long().item()
                src_list.append(int(u)); dst_list.append(int(v)); attr_list.append(int(inv_bit))

        if len(src_list) == 0:
            edge_index = torch.zeros((2,0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0,1), dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
            edge_attr = torch.tensor(attr_list, dtype=torch.long, device=device).unsqueeze(1)

        out = Data()
        out.x = template.x.clone().to(device)
        out.edge_index = edge_index
        out.edge_attr = edge_attr
        # node_depth will be computed in enforce_aig_constraints
        out.node_depth = template.node_depth.clone().to(device) if hasattr(template, 'node_depth') else None
        return out
