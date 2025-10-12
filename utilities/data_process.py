# -----------------------------
# Utilities: load/save/pad
# -----------------------------
import os
import glob
from typing import List
import torch
from torch_geometric.data import Data, Batch


def load_pt_dataset(dataset_dir, max_files=None):
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.pt")))
    if max_files is not None:
        files = files[:max_files]
    data_list = []
    for f in files:
        try:
            d = torch.load(f,weights_only=False)
            if not isinstance(d, Data):
                # Some projects save (Data, label) tuples
                if isinstance(d, (tuple, list)) and isinstance(d[0], Data):
                    d = d[0]
                else:
                    continue
            data_list.append(d)
        except Exception as e:
            print(f"[WARN] failed loading {f}: {e}")
    return data_list

def pad_data(data: Data, max_nodes: int, max_edges: int):
    """
    Pad Data to fixed size. Pads x to [max_nodes, 2].
    Pads edge_index to [2, max_edges] using -1 sentinel for unused edges.
    Pads edge_attr to [max_edges, 1].
    node_depth padded to [max_nodes].
    NOTE: This is a pragmatic approach for batched processing and gradient-penalty interpolation.
    """
    device = data.x.device if hasattr(data.x, 'device') else torch.device('cpu')
    N = data.x.size(0)
    E = data.edge_index.size(1) if (data.edge_index is not None and data.edge_index.numel()>0) else 0

    x_pad = torch.zeros((max_nodes, data.x.size(1)), dtype=data.x.dtype, device=device)
    x_pad[:N] = data.x

    node_depth_pad = torch.zeros((max_nodes,), dtype=torch.long, device=device) - 1
    if hasattr(data, 'node_depth') and data.node_depth is not None:
        nd = data.node_depth
        node_depth_pad[:nd.size(0)] = nd
    else:
        node_depth_pad[:N] = 0  # placeholder

    # pad edge_index as pairs; use -1 for empty
    ei_pad = torch.full((2, max_edges), -1, dtype=torch.long, device=device)
    ea_pad = torch.zeros((max_edges, 1), dtype=torch.long, device=device)
    if E > 0:
        cnt = min(E, max_edges)
        ei_pad[:, :cnt] = data.edge_index[:, :cnt]
        ea_pad[:cnt] = data.edge_attr[:cnt]

    out = Data()
    out.x = x_pad
    out.edge_index = ei_pad
    out.edge_attr = ea_pad
    out.node_depth = node_depth_pad
    # original sizes for later unpadding
    out._orig_n = N
    out._orig_e = E
    return out

def unpad_data(padded: Data):
    N = padded._orig_n if hasattr(padded, '_orig_n') else padded.x.size(0)
    E = padded._orig_e if hasattr(padded, '_orig_e') else (padded.edge_index.numel()//2)
    d = Data()
    d.x = padded.x[:N]
    if E > 0:
        d.edge_index = padded.edge_index[:, :E]
        d.edge_attr = padded.edge_attr[:E]
    else:
        d.edge_index = torch.zeros((2, 0), dtype=torch.long)
        d.edge_attr = torch.zeros((0,1), dtype=torch.long)
    d.node_depth = padded.node_depth[:N] if hasattr(padded, 'node_depth') else None
    return d

def save_data_list_as_pt(data_list: List[Data], out_dir: str, prefix: str = "sample", start_idx: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    for i, d in enumerate(data_list):
        path = os.path.join(out_dir, f"{prefix}_{start_idx + i:06d}.pt")
        torch.save(d, path)