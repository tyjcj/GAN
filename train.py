# train.py
# WGAN-GP + GATConv discriminator + conditional generator for AIG graphs (PyG Data)
# - Compatible with PyTorch Geometric Data objects of the form:
#   Data(x=[N,2], edge_index=[2,E], edge_attr=[E,1], node_depth=[N])
# - Usage example:
#   python train.py --dataset-dir data_files/graph/ --out-dir out_gen --epochs 10 --batch-size 1

import os
import argparse
import traceback
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

from tqdm import tqdm
import networkx as nx
from utilities.data_process import load_pt_dataset, pad_data,unpad_data,save_data_list_as_pt
from models.generator import AIGGenerator
from models.discriminator import AIGDiscriminator
from models.constraints import enforce_aig_constraints

# -----------------------------
# WGAN-GP helpers
# -----------------------------
def gradient_penalty(D, real_padded: Data, fake_padded: Data, device):
    """
    Approximate GP by interpolating node features x and edge_attr (padded arrays).
    Keep edge_index as real's edge_index (practical compromise).
    """
    alpha = torch.rand(1, 1, device=device)
    # interpolate nodes
    x_inter = (alpha * real_padded.x + (1 - alpha) * fake_padded.x).requires_grad_(True)
    # interpolate edge_attr
    ea_inter = (alpha * real_padded.edge_attr.float() + (1 - alpha) * fake_padded.edge_attr.float()).requires_grad_(True)
    # build a Data object for D
    inter = Data()
    inter.x = x_inter
    # use union edge_index: take real edge_index (padded)
    inter.edge_index = real_padded.edge_index
    inter.edge_attr = ea_inter
    # node_depth: approximate as real
    inter.node_depth = real_padded.node_depth
    out = D(inter)
    # out is scalar tensor
    grad = torch.autograd.grad(outputs=out.sum(), inputs=[x_inter, ea_inter],
                               create_graph=True, retain_graph=True, only_inputs=True)
    grad_x, grad_ea = grad
    grad_norm = torch.sqrt((grad_x.view(-1)**2).sum() + (grad_ea.view(-1)**2).sum() + 1e-12)
    gp = ((grad_norm - 1.0) ** 2)
    return gp

# -----------------------------
# Training loop
# -----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    data_list = load_pt_dataset(args.dataset_dir, max_files=args.max_files)
    if len(data_list) == 0:
        raise RuntimeError("No .pt files found in dataset_dir")
    # compute max nodes and edges if not provided
    if args.max_nodes is None:
        max_nodes = max([d.x.size(0) for d in data_list])
    else:
        max_nodes = args.max_nodes
    if args.max_edges is None:
        max_edges = max([d.edge_index.size(1) if (hasattr(d, 'edge_index') and d.edge_index is not None) else 0 for d in data_list])
    else:
        max_edges = args.max_edges

    print(f"[INFO] dataset size {len(data_list)}, max_nodes={max_nodes}, max_edges={max_edges}")
    padded_list = [pad_data(d, max_nodes=max_nodes, max_edges=max_edges) for d in data_list]

    loader = PyGDataLoader(padded_list, batch_size=args.batch_size, shuffle=True, drop_last=False)
    # models
    G = AIGGenerator(node_in_dim=2, hidden_dim=args.g_hidden, z_dim=args.z_dim, candidate_k=args.candidate_k).to(device)
    D = AIGDiscriminator(in_dim=2, hidden_dim=args.d_hidden).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))

    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch_idx, real_padded in enumerate(pbar):
            try:
                real_padded = real_padded.to(device)
                # convert padded Batch to list of templates (unpadded originals info lost but we have _orig_n)
                templates = [unpad_data(d) for d in real_padded.to_data_list()]

                # update discriminator n_critic times
                for _ in range(args.n_critic):
                    # sample batch of z per template
                    zs = [torch.randn(args.z_dim, device=device) for _ in templates]
                    raw_fakes = [G(t.to(device), z) for t, z in zip(templates, zs)]
                    # corrected fakes for D (detached)
                    corr_fakes = [enforce_aig_constraints(r) for r in raw_fakes]
                    # pad corrected fakes
                    corr_pads = [pad_data(c, max_nodes=max_nodes, max_edges=max_edges) for c in corr_fakes]
                    fake_batch_for_D = Batch.from_data_list(corr_pads).to(device)

                    # D real & fake
                    opt_D.zero_grad()
                    D_real = D(real_padded)
                    D_fake = D(fake_batch_for_D)
                    # Wasserstein loss (we want D_real - D_fake to be large)
                    loss_D = -(torch.mean(D_real) - torch.mean(D_fake))
                    # gradient penalty
                    gp = gradient_penalty(D, real_padded, fake_batch_for_D, device)
                    loss_D = loss_D + args.gp_lambda * gp
                    loss_D.backward()
                    opt_D.step()

                # ---------- train G ----------
                opt_G.zero_grad()
                # sample same zs again for consistency
                zs = [torch.randn(args.z_dim, device=device) for _ in templates]
                raw_gen = [G(t.to(device), z) for t, z in zip(templates, zs)]
                # pad raw_gen to form batch for D (note: D on raw gives gradient to G)
                raw_pads = [pad_data(r, max_nodes=max_nodes, max_edges=max_edges) for r in raw_gen]
                raw_batch = Batch.from_data_list(raw_pads).to(device)
                D_on_raw = D(raw_batch)
                # Generator wants D(raw) to be large
                loss_G_adv = -torch.mean(D_on_raw)
                # structural penalty on raw (indegree mismatch)
                loss_pen = torch.stack([compute_struct_penalty_pre_enforce(r) for r in raw_gen]).mean()
                loss_G = loss_G_adv + args.lambda_cons * loss_pen
                loss_G.backward()
                opt_G.step()

                # logging & saving
                if global_step % args.log_every == 0:
                    pbar.set_postfix({
                        'loss_D': f"{loss_D.item():.4f}",
                        'loss_G': f"{loss_G.item():.4f}",
                        'pen': f"{loss_pen.item():.4f}",
                        'gp': f"{gp.item():.4f}"
                    })
                if global_step % args.save_every == 0:
                    # save both raw and corrected for inspection
                    raw_save = [r.cpu() for r in raw_gen[:min(len(raw_gen), args.save_num)]]
                    corr_save = [enforce_aig_constraints(r).cpu() for r in raw_gen[:min(len(raw_gen), args.save_num)]]
                    save_data_list_as_pt(raw_save, args.out_dir, prefix=f"raw_ep{epoch}_st{global_step}", start_idx=0)
                    save_data_list_as_pt(corr_save, args.out_dir, prefix=f"corr_ep{epoch}_st{global_step}", start_idx=0)

                global_step += 1

            except Exception as e:
                print("[ERROR] exception in training step:", e)
                traceback.print_exc()
                continue

        # epoch checkpoint
        torch.save(G.state_dict(), os.path.join(args.out_dir, f"generator_epoch{epoch}.pt"))
        torch.save(D.state_dict(), os.path.join(args.out_dir, f"discriminator_epoch{epoch}.pt"))
        print(f"[INFO] saved checkpoints epoch {epoch}")

    print("[INFO] training finished")

# -----------------------------
# Structural penalty (pre-enforce)
# -----------------------------
def compute_struct_penalty_pre_enforce(raw: Data):
    """
    Penalize indegree mismatches on raw generated graph:
    - PIs should have indeg 0 (if generator tried to add, penalize)
    - POs indeg ideally 1
    - AND indeg ideally 2
    Works on raw.Data (no padding), returns scalar tensor
    """
    device = raw.x.device
    num_nodes = raw.x.size(0)
    indeg = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if raw.edge_index is not None and raw.edge_index.numel()>0:
        for v in raw.edge_index[1].tolist():
            indeg[v] += 1
    node_types = raw.x[:,0].long()
    loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    for i in range(num_nodes):
        t = int(node_types[i].item())
        if t == 0:
            loss = loss + indeg[i].float()
        elif t == 1:
            loss = loss + torch.clamp(indeg[i].float() - 1.0, min=0.0)
        elif t == 2:
            loss = loss + torch.clamp(indeg[i].float() - 2.0, min=0.0)
    return loss

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="out_gen")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--g-hidden", type=int, default=128)
    parser.add_argument("--d-hidden", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--candidate-k", type=int, default=512)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--lambda-cons", type=float, default=1e-3)
    parser.add_argument("--gp-lambda", type=float, default=10.0)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--save-num", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    train(args)
"""
python train.py \
  --dataset-dir data_files/graph/ \
  --out-dir results/out_gen \
  --epochs 5 \
  --batch-size 1 \
  --max-nodes 2000 \
  --candidate-k 256

"""