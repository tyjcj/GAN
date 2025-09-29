# GAN/utils_aig.py
import os
import torch
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Optional



def pad_data(data: Data, max_nodes: int) -> Data:
    """
    Pad Data.x to max_nodes and keep node_depth as well.
    Edge indices remain unchanged (they reference original node indices).
    """
    x = data.x
    n = x.size(0)
    if n < max_nodes:
        pad_len = max_nodes - n
        pad_x = torch.zeros((pad_len, x.size(1)), dtype=x.dtype)
        data.x = torch.cat([x, pad_x], dim=0)
        # node_depth pad with -1 (unconnected)
        if hasattr(data, 'node_depth'):
            nd = data.node_depth
            pad_depth = torch.full((pad_len,), -1, dtype=nd.dtype)
            data.node_depth = torch.cat([nd, pad_depth], dim=0)
    return data


def batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return batch.to(device)


def save_data_list_as_pt(lst: List[Data], out_dir: str, prefix: str = "gen", start_idx: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    for i, d in enumerate(lst):
        path = os.path.join(out_dir, f"{prefix}_{start_idx + i:06d}.pt")
        torch.save(d, path)


def load_pt_dataset(directory: str, max_files: Optional[int] = None) -> List[Data]:
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")])
    if max_files:
        files = files[:max_files]
    data_list = [torch.load(p,weights_only=False) for p in files]
    return data_list


def compute_inverted_count_per_node(edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    edge_attr shape: [E, 1] with 0/1 inversion bits
    Returns tensor shape [num_nodes] count of inverted incoming edges per node.
    """
    device = edge_index.device
    inv_counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if edge_index.numel() == 0:
        return inv_counts
    dst = edge_index[1]
    invs = edge_attr.view(-1).long()
    for d, inv in zip(dst.tolist(), invs.tolist()):
        inv_counts[d] += int(inv)
    return inv_counts


def make_batch_from_list(data_list: List[Data]) -> Batch:
    return Batch.from_data_list(data_list)
