import os
import argparse
import traceback
import csv
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utilities.data_process import load_pt_dataset, pad_data, unpad_data, save_data_list_as_pt
from models.generator import AIGGenerator
from models.discriminator import AIGDiscriminator
from models.constraints import enforce_aig_constraints, aig_constraint_penalties, validate_strict_aig



# -----------------------------
# 记录与画图（多曲线同图）
# -----------------------------
class MetricsLogger:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.rows = []

    def log(self, step, epoch, d_real, d_fake, gap, grad_norm, loss_d, loss_g, gp, pen):
        self.rows.append({
            "step": float(step),
            "epoch": int(epoch),
            "D_real": float(d_real),
            "D_fake": float(d_fake),
            "gap": float(gap),
            "grad_norm": float(grad_norm),
            "loss_D": float(loss_d),
            "loss_G": float(loss_g),
            "gp": float(gp),
            "pen": float(pen),
        })

    def flush_csv(self):
        if not self.rows:
            return
        path = os.path.join(self.out_dir, "metrics.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)

    def _plot_overlay(self, xs, ys_list, labels, title, ylabel, fname):
        plt.figure()
        for ys, lab in zip(ys_list, labels):
            plt.plot(xs, ys, label=lab)
        plt.title(title)
        plt.xlabel("step")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, fname))
        plt.close()

    def _plot_dual_axis(self, xs, ys_left, ys_right, label_left, label_right, title, fname):
        plt.figure()
        ax1 = plt.gca()
        l1, = ax1.plot(xs, ys_left, label=label_left)
        ax1.set_xlabel("step")
        ax1.set_ylabel(label_left)
        ax2 = ax1.twinx()
        l2, = ax2.plot(xs, ys_right, label=label_right, linestyle="--")
        ax2.set_ylabel(label_right)
        plt.title(title)
        ax1.legend([l1, l2], [label_left, label_right], loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, fname))
        plt.close()

    def plot_all(self):
        if not self.rows:
            return
        xs = [r["step"] for r in self.rows]
        self._plot_overlay(xs,
                           [[r["loss_D"] for r in self.rows],
                            [r["loss_G"] for r in self.rows]],
                           ["loss_D", "loss_G"],
                           "Losses", "loss", "losses.png")
        self._plot_overlay(xs,
                           [[r["D_real"] for r in self.rows],
                            [r["D_fake"] for r in self.rows]],
                           ["D_real", "D_fake"],
                           "Discriminator scores", "score", "scores.png")
        self._plot_dual_axis(xs,
                             [r["gap"] for r in self.rows],
                             [r["grad_norm"] for r in self.rows],
                             "gap (D_real - D_fake)", "grad_norm",
                             "Gap & Grad-norm", "gap_grad.png")
        self._plot_overlay(xs,
                           [[r["gp"] for r in self.rows],
                            [r["pen"] for r in self.rows]],
                           ["gp", "pen"],
                           "GP & Structure penalty", "value", "gp_pen.png")


# -----------------------------
# 工具函数
# -----------------------------
def sanitize_for_D(padded: Data) -> Data:
    """把 -1 边裁掉，让判别器输入干净。"""
    d = Data()
    d.x = padded.x
    if padded.edge_index is not None and padded.edge_index.numel() > 0:
        mask = (padded.edge_index.min(dim=0).values >= 0)
        d.edge_index = padded.edge_index[:, mask]
        if hasattr(padded, "edge_attr") and padded.edge_attr is not None and padded.edge_attr.numel() > 0:
            d.edge_attr = padded.edge_attr[mask]
        else:
            d.edge_attr = torch.zeros((d.edge_index.size(1), 1), dtype=torch.long, device=padded.x.device)
    else:
        d.edge_index = torch.zeros((2, 0), dtype=torch.long, device=padded.x.device)
        d.edge_attr = torch.zeros((0, 1), dtype=torch.long, device=padded.x.device)
    d.node_depth = padded.node_depth
    return d


def valid_edge_mask(d: Data):
    """返回 Data 中有效边的掩码（非 -1 索引）。"""
    if d.edge_index is None or d.edge_index.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=d.x.device)
    return (d.edge_index.min(dim=0).values >= 0)


def gradient_penalty(D, real_pad: Data, fake_pad: Data, device, use_edges: bool = True):
    """
    GP 只在“pad后”的 batch 上做插值，然后对 real/fake 的“共同有效边”做掩码，确保形状一致。
    返回：(gp_term, grad_norm_scalar)
    """
    # 共同有效边掩码（两侧都有效的边位）
    mask_real = valid_edge_mask(real_pad)
    mask_fake = valid_edge_mask(fake_pad)
    if (mask_real.numel() == 0 or mask_fake.numel() == 0) or (not use_edges):
        # 没有边：只对节点特征做 GP
        alpha = torch.rand(1, 1, device=device)
        x_inter = (alpha * real_pad.x + (1 - alpha) * fake_pad.x).requires_grad_(True)
        inter = Data(x=x_inter, edge_index=torch.zeros((2,0), dtype=torch.long, device=device),
                     edge_attr=torch.zeros((0,1), dtype=torch.float32, device=device),
                     node_depth=real_pad.node_depth)
        out = D(inter)
        grads = torch.autograd.grad(out.sum(), [x_inter], create_graph=True, retain_graph=True, only_inputs=True)
        grad_norm = torch.norm(grads[0].reshape(-1), p=2)
        gp = (grad_norm - 1.0) ** 2
        return gp, grad_norm.detach()

    common_mask = mask_real & mask_fake
    # 可能两个掩码没有重叠（极端情况）
    if common_mask.sum().item() == 0:
        # 退化为仅节点 GP
        alpha = torch.rand(1, 1, device=device)
        x_inter = (alpha * real_pad.x + (1 - alpha) * fake_pad.x).requires_grad_(True)
        inter = Data(x=x_inter, edge_index=torch.zeros((2,0), dtype=torch.long, device=device),
                     edge_attr=torch.zeros((0,1), dtype=torch.float32, device=device),
                     node_depth=real_pad.node_depth)
        out = D(inter)
        grads = torch.autograd.grad(out.sum(), [x_inter], create_graph=True, retain_graph=True, only_inputs=True)
        grad_norm = torch.norm(grads[0].reshape(-1), p=2)
        gp = (grad_norm - 1.0) ** 2
        return gp, grad_norm.detach()

    # 用相同 alpha 对 x / edge_attr 插值（注意两边都是“pad后”的矩阵，形状一致，随后再按 common_mask 过滤边）
    alpha = torch.rand(1, 1, device=device)
    x_inter = (alpha * real_pad.x + (1 - alpha) * fake_pad.x).requires_grad_(True)

    ea_r = real_pad.edge_attr.float()[common_mask]
    ea_f = fake_pad.edge_attr.float()[common_mask]
    ea_inter = (alpha * ea_r + (1 - alpha) * ea_f).requires_grad_(True)

    ei_inter = real_pad.edge_index[:, common_mask]  # 与 ea_inter 对齐
    inter = Data(x=x_inter, edge_index=ei_inter, edge_attr=ea_inter, node_depth=real_pad.node_depth)

    out = D(inter)
    grads = torch.autograd.grad(out.sum(), [x_inter, ea_inter], create_graph=True, retain_graph=True, only_inputs=True)
    grad_x, grad_ea = grads
    grad_norm = torch.norm(torch.cat([grad_x.reshape(-1), grad_ea.reshape(-1)], dim=0), p=2)
    gp = (grad_norm - 1.0) ** 2
    return gp, grad_norm.detach()


# -----------------------------
# 训练主循环
# -----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # 加载数据
    data_list = load_pt_dataset(args.dataset_dir, max_files=args.max_files)
    if len(data_list) == 0:
        raise RuntimeError("No .pt files found in dataset_dir")

    max_nodes = max([d.x.size(0) for d in data_list]) if args.max_nodes is None else args.max_nodes
    max_edges = max([(d.edge_index.size(1) if (hasattr(d,'edge_index') and d.edge_index is not None) else 0)
                     for d in data_list]) if args.max_edges is None else args.max_edges

    print(f"[INFO] dataset size {len(data_list)}, max_nodes={max_nodes}, max_edges={max_edges}")
    padded_list = [pad_data(d, max_nodes=max_nodes, max_edges=max_edges) for d in data_list]
    loader = PyGDataLoader(
        padded_list,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if (os.cpu_count() or 0) > 1 else False,
    )

    # 模型
    G = AIGGenerator(node_in_dim=2, hidden_dim=args.g_hidden, z_dim=args.z_dim, candidate_k=args.candidate_k).to(device)
    D = AIGDiscriminator(in_dim=2, hidden_dim=args.d_hidden).to(device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 优化器
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))

    logger = MetricsLogger(args.out_dir)
    global_step = 0

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

        # 在生成器的训练中，对生成的图进行合规性检查
        for batch_idx, real_padded in enumerate(pbar):
            try:
                real_padded = real_padded.to(device)
                templates = [unpad_data(d) for d in real_padded.to_data_list()]

                # -------- 训练 D (n_critic 次) --------
                D_real_val = D_fake_val = gap_val = gp_term_val = grad_norm_val = None
                # 仅生成一次并复用于 n_critic 循环，显著减少 Python/CPU 开销
                zs_once = [torch.randn(args.z_dim, device=device) for _ in templates]
                raw_fakes_once = [G(t.to(device), z) for t, z in zip(templates, zs_once)]
                corr_fakes_once = [enforce_aig_constraints(r) for r in raw_fakes_once]
                corr_pads_once = [pad_data(c, max_nodes=max_nodes, max_edges=max_edges) for c in corr_fakes_once]
                fake_padded_once = Batch.from_data_list(corr_pads_once).to(device)
                for _ in range(args.n_critic):
                    fake_padded = fake_padded_once

                    # 判别器前向用“裁边后”的干净图
                    real_for_D = sanitize_for_D(real_padded)
                    fake_for_D = sanitize_for_D(fake_padded)

                    opt_D.zero_grad()
                    D_real = D(real_for_D)
                    D_fake = D(fake_for_D)
                    D_real_mean = torch.mean(D_real)
                    D_fake_mean = torch.mean(D_fake)
                    gap = D_real_mean - D_fake_mean

                    # 🔧 仅节点 GP 提速
                    gp_term, grad_norm = gradient_penalty(D, real_padded, fake_padded, device, use_edges=False)

                    # 标准 WGAN-GP：min_D (E[D(fake)] - E[D(real)] + λ*GP)
                    loss_D = (D_fake_mean - D_real_mean) + args.gp_lambda * gp_term
                    loss_D.backward()
                    opt_D.step()

                    D_real_val = D_real_mean.detach()
                    D_fake_val = D_fake_mean.detach()
                    gap_val = gap.detach()
                    gp_term_val = gp_term.detach()
                    grad_norm_val = grad_norm.detach()

                # -------- 训练 G --------
                opt_G.zero_grad()
                zs = [torch.randn(args.z_dim, device=device) for _ in templates]
                raw_gen = [G(t.to(device), z) for t, z in zip(templates, zs)]

                # 对抗项：不先 enforce，鼓励 G 自学到合规结构
                raw_pads = [pad_data(r, max_nodes=max_nodes, max_edges=max_edges) for r in raw_gen]
                raw_batch = Batch.from_data_list(raw_pads).to(device)
                raw_batch_for_D = sanitize_for_D(raw_batch)
                D_on_raw = D(raw_batch_for_D)
                loss_G_adv = -torch.mean(D_on_raw)

                # 结构惩罚（GPU化）；严格校验降频到保存步
                pen_terms = [aig_constraint_penalties(r) for r in raw_gen]
                loss_pen = torch.stack([pt["total"] for pt in pen_terms]).mean()
                if (global_step % args.save_every) == 0:
                    hard_pen_scale = []
                    for r in raw_gen:
                        violations = validate_strict_aig(r, check_all_on_path=True)
                        scale = 10.0 if len(violations) > 0 else 1.0
                        hard_pen_scale.append(torch.tensor(scale, dtype=torch.float32, device=device))
                    hard_pen_scale = torch.stack(hard_pen_scale).mean()
                    loss_pen = loss_pen * hard_pen_scale
                loss_G = loss_G_adv + args.lambda_cons * loss_pen
                loss_G.backward()
                opt_G.step()

                logger.log(global_step, epoch,
                           d_real=D_real_val.item(),
                           d_fake=D_fake_val.item(),
                           gap=gap_val.item(),
                           grad_norm=grad_norm_val.item(),
                           loss_d=loss_D.item(),
                           loss_g=loss_G.item(),
                           gp=gp_term_val.item(),
                           pen=loss_pen.item())

                if global_step % args.log_every == 0:
                    pbar.set_postfix({
                        'loss_D': f"{loss_D.item():.4f}",
                        'loss_G': f"{loss_G.item():.4f}",
                        'D_real': f"{D_real_val.item():.3f}",
                        'D_fake': f"{D_fake_val.item():.3f}",
                        'gap': f"{gap_val.item():.3f}",
                        'grad||': f"{grad_norm_val.item():.3f}",
                        'gp': f"{gp_term_val.item():.4f}",
                        'pen': f"{loss_pen.item():.4f}"
                    })

                if global_step % args.save_every == 0:
                    raw_save = [r.cpu() for r in raw_gen[:min(len(raw_gen), args.save_num)]]
                    corr_list = [enforce_aig_constraints(r).cpu() for r in raw_gen[:min(len(raw_gen), args.save_num)]]
                    save_data_list_as_pt(raw_save, args.out_dir, prefix=f"raw_ep{epoch}_st{global_step}", start_idx=0)
                    save_data_list_as_pt(corr_list, args.out_dir, prefix=f"corr_ep{epoch}_st{global_step}", start_idx=0)

                    # 自动保存严格合规且 and>0、lev>0 的样本到 results/ISCAS85/pt_fake
                    valid_dir = os.path.join("results", "ISCAS85", "pt_fake")
                    os.makedirs(valid_dir, exist_ok=True)
                    valid_to_save = []
                    for i, c in enumerate(corr_list):
                        violations = validate_strict_aig(c, check_all_on_path=True)
                        # and>0、lev>0 检测：AND 计数与最大深度
                        and_count = 0
                        if hasattr(c, 'edge_index') and c.edge_index is not None and c.edge_index.numel() > 0:
                            # 粗略估计 AND 数：节点类型统计
                            if hasattr(c, 'x') and c.x is not None and c.x.size(1) > 0:
                                and_count = int((c.x[:,0] == 2).sum().item())
                        max_lev = int(c.node_depth.max().item()) if hasattr(c, 'node_depth') and c.node_depth is not None and c.node_depth.numel() > 0 else 0
                        if len(violations) == 0 and and_count > 0 and max_lev > 0:
                            valid_to_save.append(c)
                    if len(valid_to_save) > 0:
                        save_data_list_as_pt(valid_to_save, valid_dir, prefix=f"valid_ep{epoch}_st{global_step}", start_idx=0)

                global_step += 1

            except Exception as e:
                print("[ERROR] exception in training step:", e)
                traceback.print_exc()
                continue

        torch.save(G.state_dict(), os.path.join(args.out_dir, f"generator_epoch{epoch}.pt"))
        torch.save(D.state_dict(), os.path.join(args.out_dir, f"discriminator_epoch{epoch}.pt"))
        logger.flush_csv()
        logger.plot_all()
        print(f"[INFO] saved checkpoints & metrics epoch {epoch}")

    print("[INFO] training finished")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory containing the dataset.")
    parser.add_argument("--out-dir", type=str, default="out_gen", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--g-hidden", type=int, default=128, help="Hidden layer size for the generator.")
    parser.add_argument("--d-hidden", type=int, default=128, help="Hidden layer size for the discriminator.")
    parser.add_argument("--z-dim", type=int, default=128, help="Dimensionality of the latent variable.")
    parser.add_argument("--candidate-k", type=int, default=512, help="Number of candidate predecessors per node.")
    parser.add_argument("--max-nodes", type=int, default=None, help="Maximum number of nodes per graph.")
    parser.add_argument("--max-edges", type=int, default=None, help="Maximum number of edges per graph.")
    parser.add_argument("--lambda-cons", type=float, default=1e-2, help="Lambda for the structure penalty.")
    parser.add_argument("--gp-lambda", type=float, default=10.0, help="Lambda for gradient penalty.")
    parser.add_argument("--n-critic", type=int, default=3, help="Number of D updates per G update.")
    parser.add_argument("--save-every", type=int, default=10, help="Save samples every X steps.")
    parser.add_argument("--save-num", type=int, default=2, help="How many samples to save each checkpoint.")
    parser.add_argument("--log-every", type=int, default=1, help="Log every X steps.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files to load.")
    args = parser.parse_args()
    train(args)

"""
python train.py \
  --dataset-dir data_files/datasets/ISCAS85/graph \
  --out-dir results/ISCAS85/train_results \
  --epochs 10 \
  --batch-size 1 \
  --candidate-k 64

"""