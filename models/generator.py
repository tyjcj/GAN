# models/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import random

class AIGGenerator(nn.Module):
    """
    Generator returns a *raw* Data object (may violate hard AIG constraints).
    Training script will compute penalty on this raw output, then call enforce_aig_constraints()
    to produce corrected sample for D (detached).
    """

    def __init__(self,
                 node_in_dim: int = 2,
                 hidden_dim: int = 128,
                 z_dim: int = 128,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 max_prev_candidates: int = 4096,
                 depth_perturb: int = 2):
        super().__init__()
        self.device = device
        self.z_dim = z_dim
        self.max_prev_candidates = max_prev_candidates
        self.depth_perturb = depth_perturb

        # small graph encoder to produce node embeddings conditioned on template
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # projections for scoring
        self.src_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.tgt_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # inversion predictor
        self.inv_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, template: Data, z: torch.Tensor):
        """
        template: Data with fields x, edge_index (optional), node_depth (may be present)
        z: [z_dim] vector (1D)
        returns: raw Data (not enforced), with edge_index and edge_attr (0/1 ints), and x preserved (x[:,1] placeholder)
        """
        device = self.device if hasattr(self, 'device') else template.x.device
        x = template.x.to(device)
        node_types = x[:, 0].long()
        N = x.size(0)

        # compute node embeddings conditioned on template graph (if edges exist)
        if template.edge_index is None or template.edge_index.numel() == 0:
            # fallback: simple mapping
            h = torch.zeros((N, self.node_proj[0].in_features - self.z_dim), device=device)
            # if not enough information, use linear on x
            h = torch.cat([x.float(), torch.zeros((N, self.node_proj[0].in_features - x.size(1)), device=device)], dim=1)[:, :self.node_proj[0].in_features - self.z_dim]
        else:
            h = F.relu(self.conv1(x.float(), template.edge_index.to(device)))
            h = F.relu(self.conv2(h, template.edge_index.to(device)))

        z_expand = z.unsqueeze(0).repeat(N, 1)  # [N, z_dim]
        h = self.node_proj(torch.cat([h, z_expand], dim=1))  # [N, H]

        # Build candidate pools by depth but allow small perturbation to encourage diversity
        node_depth = None
        if hasattr(template, 'node_depth') and template.node_depth is not None:
            node_depth = template.node_depth.clone().long().tolist()
        else:
            # fallback: assume PI at indices where node_types==0
            node_depth = [0 if int(t)==0 else -1 for t in node_types.tolist()]

        depth_to_nodes = {}
        for idx, d in enumerate(node_depth):
            if d < 0:
                # unknown depth -> assign 0 or later
                continue
            depth_to_nodes.setdefault(d, []).append(idx)
        max_depth = max([d for d in node_depth if d >= 0]) if any(d >= 0 for d in node_depth) else 0

        src_list = []
        dst_list = []
        attr_list = []

        # Precompute candidate embeddings per depth
        for target_depth in range(1, max_depth + 1):
            targets = depth_to_nodes.get(target_depth, [])
            if len(targets) == 0:
                continue
            # allowed source depths: <= target_depth -1, but with perturbation allow up to +depth_perturb
            candidate_nodes = []
            min_depth = 0
            max_allowed = target_depth - 1 + self.depth_perturb
            for d in range(min_depth, max_allowed + 1):
                candidate_nodes.extend(depth_to_nodes.get(d, []))
            if len(candidate_nodes) == 0:
                continue
            # cap candidate set
            if len(candidate_nodes) > self.max_prev_candidates:
                candidate_nodes = random.sample(candidate_nodes, self.max_prev_candidates)
            cand_idx = torch.tensor(candidate_nodes, dtype=torch.long, device=device)
            cand_emb = h[cand_idx]  # [C, H]
            proj_cand = self.src_proj(cand_emb)  # [C, H]

            for v in targets:
                t = int(node_types[v].item())
                if t == 0:
                    continue
                tgt_emb = h[v:v+1]
                proj_tgt = self.tgt_proj(tgt_emb).squeeze(0)
                scores = torch.matmul(proj_cand, proj_tgt)  # [C]
                # topk selection
                k = 2 if t == 2 else 1
                k_eff = min(k, scores.numel())
                topk = torch.topk(scores, k_eff, largest=True)
                sel_indices = topk.indices.tolist()
                chosen = [candidate_nodes[i] for i in sel_indices]
                while len(chosen) < k:
                    chosen.append(random.choice(candidate_nodes))
                for u in chosen[:k]:
                    u_emb = h[u:u+1]
                    inv_logit = self.inv_mlp(torch.cat([u_emb.squeeze(0), tgt_emb.squeeze(0), z], dim=0))
                    inv_bit = (torch.sigmoid(inv_logit) > 0.5).long().item()
                    src_list.append(int(u))
                    dst_list.append(int(v))
                    attr_list.append(int(inv_bit))

        if len(src_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, 1), dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
            edge_attr = torch.tensor(attr_list, dtype=torch.long, device=device).unsqueeze(1)

        out = Data()
        out.x = template.x.clone().to(device)  # x[:,1] may be placeholder; will be recomputed later by constraints
        out.edge_index = edge_index
        out.edge_attr = edge_attr
        # We'll not set node_depth here; constraints will recompute
        out.node_depth = template.node_depth.clone().to(device) if hasattr(template, 'node_depth') else None

        return out
