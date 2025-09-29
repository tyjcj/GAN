# gan_aig_generator.py
import argparse
import os
import traceback
from tqdm import tqdm

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

from models.generator import AIGGenerator
from models.discriminator import AIGDiscriminator
from models.utils_aig import load_pt_dataset, pad_data, save_data_list_as_pt
from models.constraints import enforce_aig_constraints, topological_sort_kahn

def collate_and_pad(data_list, max_nodes):
    out = []
    for d in data_list:
        out.append(pad_data(d, max_nodes))
    return out

def compute_constraint_violation_loss_from_raw(raw_data: Data) -> torch.Tensor:
    # same function as before, but specifically accepts raw (pre-fix) Data
    device = raw_data.x.device
    num_nodes = raw_data.x.size(0)
    indeg = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if raw_data.edge_index is not None and raw_data.edge_index.numel() > 0:
        for v in raw_data.edge_index[1].tolist():
            indeg[v] += 1
    node_types = raw_data.x[:, 0].long()
    loss = torch.tensor(0.0, device=device)
    for i in range(num_nodes):
        t = int(node_types[i].item())
        if t == 0:
            loss += indeg[i].float()
        elif t == 1:
            loss += torch.clamp(indeg[i].float() - 1, min=0.0)
        elif t == 2:
            loss += torch.clamp(indeg[i].float() - 2, min=0.0)
    return loss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading dataset from", args.dataset_dir)
    data_list = load_pt_dataset(args.dataset_dir, max_files=args.max_files)
    if len(data_list) == 0:
        raise RuntimeError("No .pt files found in dataset_dir")

    if args.max_nodes is None:
        max_nodes = max([d.x.size(0) for d in data_list])
    else:
        max_nodes = args.max_nodes
    print(f"[INFO] Using max_nodes = {max_nodes}")

    data_list = collate_and_pad(data_list, max_nodes)
    print(f"[DEBUG] Loaded dataset: {len(data_list)} samples")

    loader = PyGDataLoader(data_list, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    print(f"[DEBUG] Loader batches: {len(loader)}")
    for i, d in enumerate(data_list[:3]):
        print(f"[DEBUG] Sample {i}: x={d.x.shape}, edge_index={d.edge_index.shape}, edge_attr={d.edge_attr.shape}, node_depth={d.node_depth.shape}")

    # Models
    G = AIGGenerator(node_in_dim=2, hidden_dim=args.g_hidden, z_dim=args.z_dim,
                     device=device, max_prev_candidates=args.max_prev_candidates).to(device)
    D = AIGDiscriminator(node_in_dim=2, hidden_dim=args.d_hidden).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5,0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5,0.999))
    bce = nn.BCELoss()

    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch_idx, real_batch in enumerate(pbar):
            try:
                real_batch = real_batch.to(device)
                templates = real_batch.to_data_list()

                # Sample a z per template and keep them (consistency)
                zs = [torch.randn(args.z_dim, device=device) for _ in templates]

                # 1) Generate raw fakes (no enforce) -> compute penalty ON RAW
                raw_fakes = []
                for t, z in zip(templates, zs):
                    raw = G(t, z)  # returns raw Data
                    raw_fakes.append(raw)

                # compute penalty on raw fakes BEFORE any correction
                loss_pen = torch.stack([compute_constraint_violation_loss_from_raw(r) for r in raw_fakes]).mean()

                # 2) Create corrected fakes for D by enforcing constraints (detached to avoid gradient through correction)
                corrected_fakes = []
                for r in raw_fakes:
                    c = enforce_aig_constraints(r)  # in-place returns Data
                    # detach attributes to ensure D doesn't propagate to G via enforced edits
                    c.edge_index = c.edge_index.detach() if isinstance(c.edge_index, torch.Tensor) else c.edge_index
                    c.edge_attr = c.edge_attr.detach() if isinstance(c.edge_attr, torch.Tensor) else c.edge_attr
                    c.x = c.x.detach() if isinstance(c.x, torch.Tensor) else c.x
                    corrected_fakes.append(c)

                fake_batch_for_D = Batch.from_data_list([pad_data(c, max_nodes) for c in corrected_fakes]).to(device)
                # Real labels and fake labels
                real_labels = torch.ones(real_batch.num_graphs, device=device)
                fake_labels = torch.zeros(fake_batch_for_D.num_graphs, device=device)

                # ---------------- Train D ----------------
                opt_D.zero_grad()
                D_real = D(real_batch)
                D_fake = D(fake_batch_for_D)
                loss_D = bce(D_real, real_labels) + bce(D_fake, fake_labels)
                loss_D.backward()
                opt_D.step()

                # ---------------- Train G ----------------
                opt_G.zero_grad()
                # regenerate raw fakes with same zs (to build graph) and compute D on raw (so gradients flow)
                gen_list = []
                for t, z in zip(templates, zs):
                    g_raw = G(t, z)  # computational graph
                    # pad but keep raw structure for gradient flow
                    g_padded = pad_data(g_raw, max_nodes)
                    gen_list.append(g_padded)
                gen_batch = Batch.from_data_list(gen_list).to(device)
                pred_on_raw = D(gen_batch)  # D applied on raw (this provides gradient to G)
                adv_labels = torch.ones(pred_on_raw.size(0), device=device)
                loss_G_adv = bce(pred_on_raw, adv_labels)

                # Final G loss uses adv + penalty (penalty computed earlier on raw_fakes)
                loss_G = loss_G_adv + args.lambda_cons * loss_pen

                loss_G.backward()
                opt_G.step()

                # Logging & saving
                if global_step % args.log_every == 0:
                    pbar.set_postfix({"loss_D": f"{loss_D.item():.4f}", "loss_G": f"{loss_G.item():.4f}", "pen": f"{loss_pen.item():.4f}"})

                if global_step % args.save_every == 0:
                    # save both a raw and its corrected version for inspection
                    to_save_raw = [r.cpu() for r in raw_fakes[:min(len(raw_fakes), args.save_num)]]
                    to_save_corr = [c.cpu() for c in corrected_fakes[:min(len(corrected_fakes), args.save_num)]]
                    save_data_list_as_pt(to_save_raw, args.out_dir, prefix=f"raw_sample_ep{epoch}_st{global_step}", start_idx=0)
                    save_data_list_as_pt(to_save_corr, args.out_dir, prefix=f"corr_sample_ep{epoch}_st{global_step}", start_idx=0)

                global_step += 1

            except Exception as e:
                print("[ERROR] Exception during training step:")
                traceback.print_exc()
                # save failing templates
                try:
                    errdir = os.path.join(args.out_dir, "error_debug")
                    os.makedirs(errdir, exist_ok=True)
                    for idx, tmp in enumerate(templates[:min(4,len(templates))]):
                        torch.save(tmp.cpu(), os.path.join(errdir, f"err_template_ep{epoch}_b{batch_idx}_{idx}.pt"))
                    print("[INFO] Saved failing templates to", errdir)
                except Exception:
                    pass
                continue

        # epoch end checkpoint
        torch.save(G.state_dict(), os.path.join(args.out_dir, f"generator_epoch{epoch}.pt"))
        torch.save(D.state_dict(), os.path.join(args.out_dir, f"discriminator_epoch{epoch}.pt"))
        print("[INFO] Saved checkpoints for epoch", epoch)

    print("[INFO] Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="out_gen")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--g-hidden", type=int, default=128)
    parser.add_argument("--d-hidden", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--max-prev-candidates", type=int, default=4096)
    parser.add_argument("--lambda-cons", type=float, default=1e-3)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--save-num", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--debug-every", type=int, default=50)
    args = parser.parse_args()
    train(args)

"""
python gan_aig_generator.py \
--dataset-dir data_files/ISCAS85/graph --out-dir results/ISCAS85 --batch-size 2
"""