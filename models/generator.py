# GAN/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from typing import Optional, List
import random
from .constraints import enforce_aig_constraints
from .utils_aig import compute_inverted_count_per_node

class AIGGenerator(nn.Module):
    """
    Conditional Generator:
    - input: template_data: Data (x=[N,2], edge_index, edge_attr, node_depth), optional
             noise z: [z_dim]
    - output: Data with same N (by default), but with newly generated edge_index & edge_attr.
    Design decision: We preserve the node set and node_depth from template to simplify ensuring DAG property;
    generator focuses on generating new wiring (edges) + edge inversion bits.
    """

    def __init__(self,
                 node_in_dim: int = 2,
                 hidden_dim: int = 128,
                 z_dim: int = 128,
                 device: torch.device = torch.device('cpu'),
                 max_prev_candidates: int = 4096):
        super().__init__()
        self.device = device
        self.z_dim = z_dim
        self.max_prev_candidates = max_prev_candidates

        # encoder for template graph -> node embeddings
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # for pair scoring (source, target) -> higher score => more likely to be selected as predecessor
        self.src_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.tgt_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # edge inversion predictor (given src embedding and tgt embedding)
        self.inv_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # logit
        )

    def forward(self, template: Data, z: Optional[torch.Tensor] = None,
                sample_hard: bool = True) -> Data:
        """
        template: Data with x [N,2], edge_index, node_depth (must exist)
        z: optional noise vector (z_dim,) or None - will be sampled internally
        sample_hard: if True, returns discrete edges; if False, returns soft scores (not used later)
        """
        assert hasattr(template, 'node_depth'), "template must have node_depth field"
        device = self.device if hasattr(self, 'device') else template.x.device
        x = template.x.to(device)
        edge_index = template.edge_index.to(device) if template.edge_index is not None else None
        node_depth = template.node_depth.to(device)

        N = x.size(0)
        if z is None:
            z = torch.randn(self.z_dim, device=device)
        if z.dim() == 1:
            z = z.unsqueeze(0)  # [1, z_dim]

        # 1) encode template graph into node embeddings
        # If template has no edges, run convs with self loops avoided
        if edge_index is None or edge_index.numel() == 0:
            # fallback: simple MLP on node features
            h = F.relu(self.node_mlp(torch.cat([x.float(), z.repeat(N, 1)], dim=1)))
        else:
            h = F.relu(self.conv1(x.float(), edge_index))
            h = F.relu(self.conv2(h, edge_index))
            # incorporate noise
            z_expand = z.repeat(N, 1)  # [N, z_dim]
            h = self.node_mlp(torch.cat([h, z_expand], dim=1))  # [N, hidden]

        # We preserve PI and PO nodes as in template for safety, only re-wire edges
        node_types = template.x[:, 0].long().to(device)  # 0,1,2

        # For each node v with type==2 (AND) we pick exactly 2 predecessors from candidates with lower depth
        # For each node v with type==1 (PO) pick exactly 1 predecessor from candidates of lower depth
        src_list = []
        dst_list = []
        attr_list = []

        # Precompute nodes per depth
        depths = node_depth.tolist()
        depth_to_nodes = {}
        for idx, d in enumerate(depths):
            if d < 0:
                continue
            depth_to_nodes.setdefault(d, []).append(idx)
        max_depth = max([d for d in depths if d >= 0]) if any(d >= 0 for d in depths) else 0

        # For speed: we'll iterate depth by depth: targets at depth d can only choose sources from depths < d
        for d in range(1, max_depth + 1):
            targets = depth_to_nodes.get(d, [])
            if len(targets) == 0:
                continue
            # candidate pool: all nodes with depth < d
            candidates = []
            for dd in range(0, d):
                candidates.extend(depth_to_nodes.get(dd, []))
            if len(candidates) == 0:
                continue

            # cap candidates to avoid O(N^2) blowup
            if len(candidates) > self.max_prev_candidates:
                # sample subset
                candidates = random.sample(candidates, self.max_prev_candidates)

            cand_idx = torch.tensor(candidates, dtype=torch.long, device=device)
            cand_emb = h[cand_idx]  # [C, H]

            # precompute projections
            proj_cand = self.src_proj(cand_emb)  # [C, H]

            for v in targets:
                t = int(node_types[v].item())
                if t == 0:
                    continue  # PI nodes must have no predecessors
                tgt_emb = h[v:v+1]  # [1,H]
                proj_tgt = self.tgt_proj(tgt_emb).squeeze(0)  # [H]

                # similarity scores
                # scores = proj_cand @ proj_tgt  => [C]
                scores = torch.matmul(proj_cand, proj_tgt)  # [C]
                # mask self if present in candidates
                # take top-k depending on node type
                if t == 2:
                    k = 2
                else:
                    k = 1
                # if candidates < k, just allow duplicates (will be fixed later)
                k_eff = min(k, scores.size(0))
                topk = torch.topk(scores, k_eff, largest=True)
                sel_idx = topk.indices.tolist()

                # map back to node indices
                chosen = [candidates[i] for i in sel_idx]
                # if fewer than needed, pad with random others
                while len(chosen) < k:
                    chosen.append(random.choice(candidates))
                # produce inversion bits
                for u in chosen[:k]:
                    u_emb = h[u:u+1]  # [1,H]
                    inv_logit = self.inv_mlp(torch.cat([u_emb.squeeze(0), tgt_emb.squeeze(0), z.squeeze(0)], dim=0))
                    inv_bit = (torch.sigmoid(inv_logit) > 0.5).long().item() if sample_hard else torch.sigmoid(inv_logit).item()
                    src_list.append(int(u))
                    dst_list.append(int(v))
                    attr_list.append(int(inv_bit))

        # After building edges, pack them into Data
        if len(src_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, 1), dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
            edge_attr = torch.tensor(attr_list, dtype=torch.long, device=device).unsqueeze(1)

        out = Data()
        out.x = template.x.clone().to(device)  # preserve node type and placeholder inverted count (will be set by constraints)
        out.edge_index = edge_index
        out.edge_attr = edge_attr
        out.node_depth = template.node_depth.clone().to(device)

        # hard enforce constraints to guarantee AIG validity
        out = enforce_aig_constraints(out, max_prev_candidates=self.max_prev_candidates, allow_fix_cycles=True)

        return out
