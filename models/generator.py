# models/generator.py
import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import List, Optional

PI, PO, AND = 0, 1, 2


class AIGGenerator(nn.Module):
    """
    DAG-safe AIG 生成器

    输入:
      - template: Data，x[:,0]∈{0,1,2} 为 {PI,PO,AND}，x[:,1] 为反相入边计数；可选 node_depth
      - z: (z_dim,) 或 (1, z_dim)

    约束:
      - 仅从“已处理节点 seen”连到当前目标 v  ->  无回边、无自环  (DAG)
      - 不从 PO 出边；不指向 PI
      - AND 入度=2、PO 入度=1；候选不足会复制首选边补齐
      - edge_attr 为反相标志位 {0,1}
      - 输出 x[:,1] 会被重写为每个节点的“反相入边计数”
    """
    def __init__(
        self,
        node_in_dim: int = 2,
        hidden_dim: int = 128,
        z_dim: int = 128,
        candidate_k: int = 512,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.candidate_k = candidate_k
        self.device = device

        # 节点特征编码
        self.node_enc = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 融合全局噪声 z 的节点表示
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 边存在性打分: [h_src, h_tgt, z] -> 标量 logit
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 反相比特预测: [h_src, h_tgt, z] -> 标量 logit (sigmoid > 0.5 判 1)
        self.edge_attr_pred = nn.Sequential(
            nn.Linear(hidden_dim * 2 + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _build_order(types: torch.Tensor, node_depth: Optional[torch.Tensor]) -> List[int]:
        """
        生成处理顺序: 优先按 node_depth 升序；否则 PI 视作 0、其余按索引。
        保证“先处理的节点”只能连向“后处理的节点”，从而天然无回边/无自环。
        """
        N = types.size(0)
        if node_depth is not None and node_depth.numel() == N:
            depth = node_depth.detach().cpu().tolist()
        else:
            depth = [(0 if int(types[i].item()) == PI else i) for i in range(N)]
        order = list(range(N))
        order.sort(key=lambda i: (depth[i], i))
        return order

    def _score_candidates(
        self,
        h_cond: torch.Tensor,          # (N, H)
        z: torch.Tensor,               # (z_dim,)
        src_idx: torch.Tensor,         # (C,)
        tgt_idx: int,                  # 标量
    ):
        """
        候选源 src_idx -> 目标 tgt_idx 的存在性打分 logits (C,) 与拼接特征 edge_in (C, 2H+z)。
        """
        cand_h = h_cond[src_idx]                               # (C,H)
        tgt_h  = h_cond[tgt_idx:tgt_idx+1].expand_as(cand_h)   # (C,H)
        z_rep  = z.unsqueeze(0).expand(cand_h.size(0), -1)     # (C,z)
        edge_in = torch.cat([cand_h, tgt_h, z_rep], dim=1)     # (C,2H+z)
        logits  = self.edge_scorer(edge_in).view(-1)           # (C,)
        return logits, edge_in

    def forward(self, template: Data, z: torch.Tensor) -> Data:
        device = template.x.device if hasattr(template, "x") else self.device

        # 标准化 z 形状
        if z.dim() == 2 and z.size(0) == 1:
            z = z.squeeze(0)
        z = z.to(device)

        # 节点与类型
        x_in  = template.x.to(device).float()             # (N,2)
        types = template.x[:, 0].long().to(device)        # (N,)
        N = types.size(0)

        # 编码 + 条件
        h = self.node_enc(x_in)                           # (N,H)
        z_expand = z.unsqueeze(0).expand(N, -1)           # (N,z)
        h_cond = self.node_proj(torch.cat([h, z_expand], dim=1))  # (N,H)

        # 生成顺序 (DAG)
        node_depth: Optional[torch.Tensor] = getattr(template, "node_depth", None)
        node_depth = node_depth.to(device) if node_depth is not None else None
        order = self._build_order(types, node_depth)

        # seen 集合：已处理节点掩码；idx_all 用于构造布尔筛选
        idx_all = torch.arange(N, device=device)
        seen_mask = torch.zeros(N, dtype=torch.bool, device=device)

        # 累积边(分块，最后一次 cat)
        src_chunks, dst_chunks, inv_chunks = [], [], []

        for v in order:
            t_v = int(types[v].item())
            # PI: 作为候选源但没有入边
            if t_v == PI:
                seen_mask[v] = True
                continue

            # 候选源: 已 seen 且 非 PO 且 非 self
            cand_mask = seen_mask & (types != PO) & (idx_all != v)
            cand_idx = torch.nonzero(cand_mask, as_tuple=False).view(-1)  # (C,)

            want_k = 2 if t_v == AND else 1

            # 若没有可用候选，回退用“最近的已见节点”补齐
            if cand_idx.numel() == 0:
                seen_idx = torch.nonzero(seen_mask, as_tuple=False).view(-1)
                if seen_idx.numel() > 0:
                    # 取最后一个（也可改成优先 PI）
                    fallback_src = seen_idx[-1:].repeat(want_k)
                    src_chunks.append(fallback_src)
                    dst_chunks.append(torch.full_like(fallback_src, v))
                    inv_chunks.append(torch.zeros_like(fallback_src))  # 反相默认 0
                seen_mask[v] = True
                continue

            # 裁剪候选规模（尾部截断，可改 randperm 采样）
            if cand_idx.numel() > self.candidate_k:
                cand_idx = cand_idx[-self.candidate_k:]

            # 打分 & 选 top-k
            logits, edge_in = self._score_candidates(h_cond, z, cand_idx, v)
            k_eff = int(min(want_k, logits.numel()))
            if k_eff == 0:
                seen_mask[v] = True
                continue

            topv, topi = torch.topk(logits, k=k_eff, largest=True)
            chosen_src = cand_idx[topi]                                  # (k_eff,)
            inv_logits = self.edge_attr_pred(edge_in[topi])              # (k_eff,1)
            inv_bits = (torch.sigmoid(inv_logits).view(-1) > 0.5).long() # (k_eff,)

            # 记录
            src_v = chosen_src
            dst_v = torch.full_like(src_v, v)
            inv_v = inv_bits

            # 候选不足时复制首条补齐
            need = want_k - k_eff
            if need > 0:
                src_v = torch.cat([src_v, src_v[:1].repeat(need)], dim=0)
                dst_v = torch.cat([dst_v, dst_v[:1].repeat(need)], dim=0)
                inv_v = torch.cat([inv_v, inv_v[:1].repeat(need)], dim=0)

            src_chunks.append(src_v)
            dst_chunks.append(dst_v)
            inv_chunks.append(inv_v)

            # 标记已见
            seen_mask[v] = True

        # 组装边张量
        if len(src_chunks) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr  = torch.zeros((0, 1), dtype=torch.long, device=device)
        else:
            src_all = torch.cat(src_chunks, dim=0)
            dst_all = torch.cat(dst_chunks, dim=0)
            inv_all = torch.cat(inv_chunks, dim=0).view(-1, 1).long()
            edge_index = torch.stack([src_all, dst_all], dim=0).long()
            edge_attr  = inv_all

        # 写回 x[:,1] = 反相入边计数
        x_out = template.x.clone().to(device)  # 保留原 dtypes
        inv_count = torch.zeros(N, dtype=torch.long, device=device)
        if edge_index.numel() > 0:
            dst = edge_index[1]
            inv = edge_attr.view(-1).long()
            inv_count.scatter_add_(0, dst, (inv == 1).long())
        x_out[:, 1] = inv_count

        return Data(
            x=x_out,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_depth=(template.node_depth.clone().to(device)
                        if getattr(template, "node_depth", None) is not None else None),
        )
