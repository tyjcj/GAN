# GAN/gan_aig_generator.py
import argparse
import os
import traceback
from tqdm import tqdm

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

# 导入你工程下的模块（请保证 models/ 下存在这些文件）
from models.generator import AIGGenerator
from models.discriminator import AIGDiscriminator
from models.utils_aig import load_pt_dataset, pad_data, save_data_list_as_pt
from models.constraints import enforce_aig_constraints

# ----------------------
# helper
# ----------------------
def collate_and_pad(data_list, max_nodes):
    out = []
    for d in data_list:
        if not isinstance(d, Data):
            raise TypeError(f"Expected torch_geometric.data.Data, got {type(d)}")
        out.append(pad_data(d, max_nodes))
    return out


def compute_constraint_violation_loss(data: Data) -> torch.Tensor:
    """ 非微分的约束惩罚：惩罚 indegree 与 AIG 规则不符的节点 """
    device = data.x.device
    num_nodes = data.x.size(0)
    indeg = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if data.edge_index is not None and data.edge_index.numel() > 0:
        # edge_index shape [2, E]
        for v in data.edge_index[1].tolist():
            indeg[v] += 1
    node_types = data.x[:, 0].long()
    loss = torch.tensor(0.0, device=device)
    for i in range(num_nodes):
        t = int(node_types[i].item())
        if t == 0:   # PI: indegree 必须为0
            loss += indeg[i].float()
        elif t == 1: # PO: indegree 必须为1
            loss += torch.clamp(indeg[i].float() - 1, min=0.0)
        elif t == 2: # AND: indegree 必须为2
            loss += torch.clamp(indeg[i].float() - 2, min=0.0)
    return loss


# ----------------------
# train function
# ----------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading dataset from", args.dataset_dir)
    data_list = load_pt_dataset(args.dataset_dir, max_files=args.max_files)
    if len(data_list) == 0:
        raise RuntimeError("No .pt files found in dataset_dir")

    # determine max_nodes
    if args.max_nodes is None:
        max_nodes = max([d.x.size(0) for d in data_list])
    else:
        max_nodes = args.max_nodes
    print(f"[INFO] Using max_nodes = {max_nodes}")

    # pad dataset
    data_list = collate_and_pad(data_list, max_nodes)
    print(f"[DEBUG] Loaded dataset: {len(data_list)} samples")

    loader = PyGDataLoader(data_list, batch_size=args.batch_size, shuffle=True, drop_last=False,
                           num_workers=0)  # num_workers 可按需调整
    print(f"[DEBUG] Loader batches: {len(loader)}")

    # print a few examples
    for i, d in enumerate(data_list[:3]):
        print(f"[DEBUG] Sample {i}: x={d.x.shape}, edge_index={d.edge_index.shape}, edge_attr={d.edge_attr.shape}, node_depth={d.node_depth.shape}")

    # models
    G = AIGGenerator(node_in_dim=2, hidden_dim=args.g_hidden, z_dim=args.z_dim,
                     device=device, max_prev_candidates=args.max_prev_candidates).to(device)
    D = AIGDiscriminator(node_in_dim=2, hidden_dim=args.d_hidden).to(device)

    # optimizers & loss
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    global_step = 0

    # Training loop
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch_idx, real_batch in enumerate(pbar):
            try:
                # Ensure batch to device
                real_batch = real_batch.to(device)
                num_graphs = real_batch.num_graphs

                # Debug prints (light)
                if global_step % args.debug_every == 0:
                    print(f"\n[DEBUG] Epoch {epoch} step {global_step} - batch_idx {batch_idx} - num_graphs {num_graphs}")
                    print(f" real_batch.x.shape = {real_batch.x.shape}, edge_index.shape = {real_batch.edge_index.shape}")

                # -- Create fake samples from generator (conditioned on each sample) --
                # Convert batch to per-sample Data list (safe & reliable)
                templates = real_batch.to_data_list()
                fake_list = []
                for i, template in enumerate(templates):
                    # sample z per template
                    z = torch.randn(args.z_dim, device=device)
                    # generator returns a Data (must)
                    fake = G(template, z)
                    if not isinstance(fake, Data):
                        # try if generator returned (data, extras)
                        if isinstance(fake, (list, tuple)) and isinstance(fake[0], Data):
                            fake = fake[0]
                        else:
                            raise TypeError(f"Generator must return torch_geometric.data.Data, got {type(fake)}")

                    # enforce constraints and pad
                    fake = enforce_aig_constraints(fake)
                    fake = pad_data(fake, max_nodes)
                    fake_list.append(fake)

                # merge to batch for D
                fake_batch = Batch.from_data_list(fake_list).to(device)

                # -------------------- Train Discriminator --------------------
                opt_D.zero_grad()
                real_labels = torch.ones(real_batch.num_graphs, device=device)
                fake_labels = torch.zeros(fake_batch.num_graphs, device=device)

                print("[DEBUG] Before D forward")
                D_real = D(real_batch)
                print("[DEBUG] After D(real_batch)")

                D_fake = D(fake_batch)
                print("[DEBUG] After D(fake_batch)")

                # If D returns probabilities (0..1) use BCELoss; if logits, user should adjust
                loss_D = bce(D_real, real_labels) + bce(D_fake, fake_labels)
                loss_D.backward()
                opt_D.step()

                # -------------------- Train Generator --------------------
                opt_G.zero_grad()
                # regenerate (or reuse) fakes: safer to regenerate so gradients flow
                gen_list = []
                for i, template in enumerate(templates):
                    z = torch.randn(args.z_dim, device=device)
                    fake = G(template, z)
                    if isinstance(fake, (list, tuple)) and isinstance(fake[0], Data):
                        fake = fake[0]
                    if not isinstance(fake, Data):
                        raise TypeError(f"Generator must return torch_geometric.data.Data, got {type(fake)}")
                    fake = enforce_aig_constraints(fake)
                    fake = pad_data(fake, max_nodes)
                    gen_list.append(fake)
                gen_batch = Batch.from_data_list(gen_list).to(device)

                pred = D(gen_batch)
                adv_labels = torch.ones(pred.size(0), device=device)
                loss_G_adv = bce(pred, adv_labels)

                # penalty (non-differentiable proxy)
                loss_pen = torch.stack([compute_constraint_violation_loss(g) for g in gen_list]).mean()
                loss_G = loss_G_adv + args.lambda_cons * loss_pen

                print("[DEBUG] Before G backward")
                loss_G.backward()
                print("[DEBUG] After G backward")

                opt_G.step()

                # Logging
                if global_step % args.log_every == 0:
                    pbar.set_postfix({
                        "loss_D": f"{loss_D.item():.4f}",
                        "loss_G": f"{loss_G.item():.4f}",
                        "pen": f"{loss_pen.item():.4f}"
                    })

                # Save generated samples occasionally
                if global_step % args.save_every == 0:
                    # save first few generated graphs (cpu)
                    to_save = [g.cpu() for g in gen_list[: min(len(gen_list), args.save_num)]]
                    # filenames prefixed with step
                    save_data_list_as_pt(to_save, args.out_dir, prefix=f"sample_{epoch}_{global_step}", start_idx=0)

                global_step += 1

            except Exception as e:
                # print stack and continue training next batch (prevents hang)
                print("[ERROR] exception during training step, printing traceback:")
                traceback.print_exc()
                # optionally save failing templates to disk for inspection
                try:
                    err_dir = os.path.join(args.out_dir, "error_debug")
                    os.makedirs(err_dir, exist_ok=True)
                    for idx, tmp in enumerate(templates[: min(4, len(templates)) ]):
                        torch.save(tmp.cpu(), os.path.join(err_dir, f"err_template_ep{epoch}_b{batch_idx}_{idx}.pt"))
                    print(f"[INFO] Saved up to 4 templates to {err_dir} for debugging.")
                except Exception:
                    pass
                # continue to next batch
                continue

        # save checkpoints at epoch end
        torch.save(G.state_dict(), os.path.join(args.out_dir, f"generator_epoch{epoch}.pt"))
        torch.save(D.state_dict(), os.path.join(args.out_dir, f"discriminator_epoch{epoch}.pt"))
        print(f"[INFO] Saved checkpoints for epoch {epoch}")

    print("[INFO] Training finished.")


# ----------------------
# CLI
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True, help="dir with .pt Data files")
    parser.add_argument("--out-dir", type=str, default="out_gen", help="output dir")
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
    # logging/debug freq
    parser.add_argument("--log-every", type=int, default=1, help="how often (steps) to update tqdm postfix")
    parser.add_argument("--debug-every", type=int, default=10, help="how often to print debug banner")
    args = parser.parse_args()

    train(args)
"""
python gan_aig_generator.py \
  --dataset-dir data_files/graph/ \
  --out-dir data_files/out_gen \
  --epochs 10 \
  --batch-size 2

"""