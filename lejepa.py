import os
import json
import glob
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP
import scipy.io as sio
import numpy as np
import wandb
import hydra
import tqdm
from omegaconf import DictConfig

from ss_mamba_model import (
    block_1D,
    spectral_spatial_block,
    PatchEmbed_2D,
    PatchEmbed_Spe,
    positional_embedding_1d,
)
from pos_embed import get_2d_sincos_pos_embed

embed_dim = 256


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(all_preds, all_labels, num_classes):
    n    = len(all_labels)
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        conf[t, p] += 1

    OA = conf.diagonal().sum() / n

    per_class_acc = np.zeros(num_classes, dtype=np.float64)
    present = []
    for c in range(num_classes):
        row_sum = conf[c].sum()
        if row_sum > 0:
            per_class_acc[c] = conf[c, c] / row_sum
            present.append(c)
    AA = per_class_acc[present].mean() if present else 0.0

    row_sums = conf.sum(axis=1)
    col_sums = conf.sum(axis=0)
    p_e   = (row_sums * col_sums).sum() / (n ** 2)
    kappa = (OA - p_e) / (1.0 - p_e) if (1.0 - p_e) > 1e-12 else 0.0

    return float(OA), float(AA), float(kappa), per_class_acc, conf


def save_results(results_path, dataset_name, run_name, metrics):
    data = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            data = json.load(f)
    data.setdefault(dataset_name, []).append({"run": run_name, **metrics})
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Resume helper — finds the latest periodic epoch checkpoint for a dataset
# ---------------------------------------------------------------------------
def find_latest_epoch_ckpt(log_dir: str, dataset_name: str):
    """
    Scans log_dir for files matching  {dataset_name}_lejepa_*_ep{N}.pt
    Returns (path, epoch_completed) of the highest-N file, or (None, -1).

    epoch_completed is the epoch index stored inside the checkpoint (0-based),
    so training should resume from epoch_completed + 1.
    """
    pattern = os.path.join(log_dir, f"{dataset_name}_lejepa_*_ep*.pt")
    candidates = glob.glob(pattern)

    best_path  = None
    best_epoch = -1

    for path in candidates:
        # filename: PaviaU_lejepa_20260414_123456_ep200.pt
        m = re.search(r"_ep(\d+)\.pt$", path)
        if m:
            ep_in_filename = int(m.group(1))   # 1-based number in filename
            # load only the scalar epoch field — no need to load weights yet
            try:
                meta = torch.load(path, map_location="cpu")
                ep   = int(meta.get("epoch", ep_in_filename - 1))
            except Exception:
                ep = ep_in_filename - 1
            if ep > best_epoch:
                best_epoch = ep
                best_path  = path

    return best_path, best_epoch


# ---------------------------------------------------------------------------
# SIGReg — unchanged
# ---------------------------------------------------------------------------
class SIGReg(nn.Module):
    def __init__(self, proj_dim, knots=17, num_projections=512):
        super().__init__()
        t       = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt      = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[0] = weights[-1] = dt
        phi     = torch.exp(-t.square() / 2.0)
        self.register_buffer("t",       t)
        self.register_buffer("phi",     phi)
        self.register_buffer("weights", weights * phi)
        A = torch.randn(proj_dim, num_projections)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)
        self.register_buffer("A", A)

    def forward(self, proj):
        z        = proj @ self.A.to(proj.device)
        x_t      = z.unsqueeze(-1) * self.t
        cos_mean = torch.cos(x_t).mean(dim=-3)
        sin_mean = torch.sin(x_t).mean(dim=-3)
        err      = (cos_mean - self.phi).pow(2) + sin_mean.pow(2)
        return (err @ self.weights).mean()


# ---------------------------------------------------------------------------
# HSI_MambaEncoder — unchanged
# ---------------------------------------------------------------------------
class HSI_MambaEncoder(nn.Module):
    def __init__(
        self,
        bands: int          = 200,
        patch_size: int     = 11,
        proj_dim: int       = 128,
        hid_chans: int      = 32,
        spa_patch_size: int = 1,
        spe_patch_size: int = 2,
        center_size: int    = 3,
        depth: int          = 4,
        bi: bool            = True,
        drop_path: float    = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.bands      = bands
        self.half_spa   = patch_size  // 2
        self.half_spe   = center_size // 2

        self.dimen_redu = nn.Sequential(
            nn.Conv2d(bands, hid_chans, kernel_size=1, bias=True),
            nn.BatchNorm2d(hid_chans),
            nn.ReLU(),
            nn.Conv2d(hid_chans, hid_chans, kernel_size=1, bias=True),
            nn.BatchNorm2d(hid_chans),
        )

        self.spa_patch_embed = PatchEmbed_2D(
            img_size=(patch_size, patch_size),
            patch_size=spa_patch_size,
            in_chans=hid_chans,
            embed_dim=embed_dim,
        )
        spa_num_patches = self.spa_patch_embed.num_patches

        self.spe_patch_embed = PatchEmbed_Spe(
            img_size=(center_size, center_size),
            patch_size=spe_patch_size,
            embed_dim=embed_dim,
        )
        spe_num_patches = bands // spe_patch_size

        self.spa_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spe_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spa_pos_embed = nn.Parameter(
            torch.zeros(1, spa_num_patches + 1, embed_dim), requires_grad=False
        )
        self.spe_pos_embed = nn.Parameter(
            positional_embedding_1d(spe_num_patches + 1, embed_dim), requires_grad=False
        )

        N = spa_num_patches + spe_num_patches + 2

        self.ss_blocks = nn.ModuleList([
            spectral_spatial_block(
                embed_dim=embed_dim, bi=bi, N=N,
                drop_path=drop_path, cls=True, fu=True,
            )
            for _ in range(depth)
        ])

        self.spa_norm    = nn.LayerNorm(embed_dim)
        self.spe_norm    = nn.LayerNorm(embed_dim)
        self.branch_gate = nn.Parameter(torch.zeros(embed_dim))
        self.proj        = MLP(embed_dim, [1024, 1024, proj_dim], norm_layer=nn.BatchNorm1d)
        self._init_weights()

    def _init_weights(self):
        grid_size = int(self.spa_patch_embed.num_patches ** 0.5)
        spa_pos   = get_2d_sincos_pos_embed(self.spa_pos_embed.shape[-1], grid_size, cls_token=True)
        self.spa_pos_embed.data.copy_(torch.from_numpy(spa_pos).float().unsqueeze(0))

    def forward(self, x: torch.Tensor):
        N, V, B, H, W = x.shape
        x  = x.flatten(0, 1)
        NV = N * V

        x_spa = self.dimen_redu(x)
        x_spa = self.spa_patch_embed(x_spa)
        x_spa = x_spa + self.spa_pos_embed[:, :-1, :]
        spa_cls = (self.spa_cls_token + self.spa_pos_embed[:, -1:, :]).expand(NV, -1, -1)
        x_spa   = torch.cat([x_spa, spa_cls], dim=1)

        cx, cs  = self.half_spa, self.half_spe
        x_ctr   = x[:, :, cx - cs : cx + cs + 1, cx - cs : cx + cs + 1]
        x_spe   = self.spe_patch_embed(x_ctr)
        x_spe   = x_spe + self.spe_pos_embed[:, :-1, :]
        spe_cls = (self.spe_cls_token + self.spe_pos_embed[:, -1:, :]).expand(NV, -1, -1)
        x_spe   = torch.cat([x_spe, spe_cls], dim=1)

        for blk in self.ss_blocks:
            x_spa, x_spe = blk(x_spa, x_spe)

        x_spa = self.spa_norm(x_spa)
        x_spe = self.spe_norm(x_spe)

        g   = torch.sigmoid(self.branch_gate)
        emb = g * x_spa[:, -1] + (1 - g) * x_spe[:, -1]

        proj = self.proj(emb)
        proj = proj.reshape(N, V, -1).transpose(0, 1)
        return emb, proj


# ---------------------------------------------------------------------------
# HSI_Dataset
# Augmentation uses pure torch ops — no torchvision v2 spatial transforms.
# v2 Beta misinterprets (220, 11, 11) tensors and silently crops spatial dims.
# ---------------------------------------------------------------------------
class HSI_Dataset(Dataset):
    def __init__(self, cube_path, gt_path, split="train", patch_size=11, V=1, train_ratio=0.6, min_train_per_class=5):
        self.V     = V
        self.patch = patch_size
        self.dataset_name = os.path.splitext(os.path.basename(cube_path))[0]

        cube = sio.loadmat(cube_path)
        gt   = sio.loadmat(gt_path)
        cube = next(v for v in cube.values() if isinstance(v, np.ndarray) and v.ndim == 3)
        gt   = next(v for v in gt.values()   if isinstance(v, np.ndarray) and v.ndim == 2)

        cube       = cube.astype(np.float32)
        H, W, B    = cube.shape
        self.bands = B

        cube = (cube - cube.mean(axis=(0, 1))) / (cube.std(axis=(0, 1)) + 1e-6)
        pad  = patch_size // 2
        cube = np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

        all_coords = np.argwhere(gt > 0)
        all_labels = gt[gt > 0] - 1
        self.classes = int(all_labels.max() + 1)

        # --- Stratified split: sample train_ratio per class ---
        np.random.seed(0)
        train_idx, test_idx = [], []

        for c in range(self.classes):
            cls_idx = np.where(all_labels == c)[0]
            np.random.shuffle(cls_idx)

            n_train = max(min_train_per_class, int(len(cls_idx) * train_ratio))
            n_train = min(n_train, len(cls_idx))  # can't exceed available

            train_idx.append(cls_idx[:n_train])
            test_idx.append(cls_idx[n_train:])

        train_idx = np.concatenate(train_idx)
        test_idx  = np.concatenate(test_idx)

        if split == "train":
            sel = train_idx
        else:
            sel = test_idx

        self.coords = all_coords[sel]
        self.labels = all_labels[sel]
        self.cube   = cube
        self.pad    = pad

    # ---- spatial augmentations using torch — size-safe -------------------
    def rand_hflip(self, p):
        return torch.flip(p, dims=[-1]) if torch.rand(1).item() > 0.5 else p

    def rand_vflip(self, p):
        return torch.flip(p, dims=[-2]) if torch.rand(1).item() > 0.5 else p

    def rand_rot90(self, p):
        return torch.rot90(p, k=torch.randint(4, (1,)).item(), dims=[-2, -1])

    # ---- spectral augmentations ------------------------------------------
    def spectral_jitter(self, p):
        return p * (1 + 0.02 * torch.randn(1, 1, 1))
    def band_dropout(self, p):
        return p * (torch.rand(p.shape[0]) > 0.1).float()[:, None, None]
    def spectral_smooth(self, p):
        return (p + torch.roll(p, 1, 0) + torch.roll(p, -1, 0)) / 3.0
    def band_scaling(self, p):
        return p * (1 + 0.05 * torch.randn(p.shape[0]))[:, None, None]
    def spectral_mask(self, p):
        B = p.shape[0]
        w = torch.randint(5, 15, (1,)).item()
        s = torch.randint(0, B - w, (1,)).item()
        p = p.clone(); p[s:s + w] = 0; return p
    def spatial_cutout(self, p):
        _, H, W = p.shape
        h = torch.randint(H // 4, H // 2, (1,)).item()
        w = torch.randint(W // 4, W // 2, (1,)).item()
        y = torch.randint(0, H - h, (1,)).item()
        x = torch.randint(0, W - w, (1,)).item()
        p = p.clone(); p[:, y:y + h, x:x + w] = 0; return p
    def spectral_noise(self, p):
        return p + 0.01 * torch.randn_like(p)

    def augment(self, p):
        """All augmentations chained — pure torch, guaranteed size-preserving."""
        p = self.rand_hflip(p)
        p = self.rand_vflip(p)
        p = self.rand_rot90(p)
        p = self.spectral_jitter(p)
        p = self.band_scaling(p)
        p = self.band_dropout(p)
        p = self.spectral_mask(p)
        p = self.spatial_cutout(p)
        p = self.spectral_smooth(p)
        p = self.spectral_noise(p)
        return p

    # ---- dataset interface -----------------------------------------------
    def __len__(self):
        return len(self.coords)

    def extract_patch(self, y, x):
        p = self.patch
        return torch.tensor(self.cube[y:y + p, x:x + p]).permute(2, 0, 1).clone()

    def __getitem__(self, i):
        y, x  = self.coords[i]
        label = self.labels[i]
        views = [self.augment(self.extract_patch(y, x)) if self.V > 1
                 else self.extract_patch(y, x)
                 for _ in range(self.V)]
        return torch.stack(views), label


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
@hydra.main(version_base=None)
def main(cfg: DictConfig):
    import time
    log_dir      = "logs"
    os.makedirs(log_dir, exist_ok=True)
    results_path = os.path.join(log_dir, "results.json")

    cube_path = getattr(cfg, "cube_path", "data/Indian_pines.mat")
    gt_path   = getattr(cfg, "gt_path",   "data/Indian_pines_gt.mat")

    train_ds = HSI_Dataset(cube_path, gt_path, split="train", V=cfg.V)
    test_ds  = HSI_Dataset(cube_path, gt_path, split="val",   V=1)
    dataset_name = train_ds.dataset_name

    train = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True,
                       drop_last=False, num_workers=0)
    test  = DataLoader(test_ds,  batch_size=256, num_workers=4)

    print(f"Dataset : {dataset_name}  |  classes: {train_ds.classes}  |  bands: {train_ds.bands}")
    print(cfg)

    # ---- build all modules (always, regardless of resume) ----------------
    net = HSI_MambaEncoder(
        bands=train_ds.bands,
        patch_size=train_ds.patch,
        proj_dim=cfg.proj_dim,
        hid_chans=getattr(cfg, "hid_chans",       32),
        spa_patch_size=getattr(cfg, "spa_patch_size", 1),
        spe_patch_size=getattr(cfg, "spe_patch_size", 5),
        center_size=getattr(cfg, "center_size",     3),
        depth=getattr(cfg, "depth",                 4),
        bi=getattr(cfg, "bi",                       True),
    ).to("cuda")

    class MLPProbe(nn.Module):
        def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.1):
            super().__init__()
            self.norm1 = nn.LayerNorm(in_dim)
            self.fc1   = nn.Linear(in_dim, in_dim)
            self.fc2   = nn.Linear(in_dim, in_dim)
            self.norm2 = nn.LayerNorm(in_dim)
            self.head  = nn.Linear(in_dim, num_classes)
            self.drop  = nn.Dropout(dropout)

        def forward(self, x):
            x = self.norm1(x)
            h = self.drop(F.gelu(self.fc1(x)))
            h = self.drop(F.gelu(self.fc2(h)))
            x = self.norm2(x + h)
            return self.head(x)

    probe  = MLPProbe(embed_dim, train_ds.classes).to("cuda")
    sigreg = SIGReg(proj_dim=cfg.proj_dim).to("cuda")

    gate_params  = [p for n, p in net.named_parameters() if n == "branch_gate"]
    other_params = [p for n, p in net.named_parameters() if n != "branch_gate"]

    g1 = {"params": other_params,         "lr": cfg.lr, "weight_decay": 1e-2}
    g2 = {"params": probe.parameters(),   "lr": 3e-3,   "weight_decay": 1e-7}
    g3 = {"params": gate_params,          "lr": 1e-3,   "weight_decay": 0.0}
    opt = torch.optim.AdamW([g1, g2, g3])

    warmup_steps = len(train) * 5
    total_steps  = len(train) * cfg.epochs
    s1           = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2           = CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps), eta_min=1e-5)
    scheduler    = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler  = GradScaler(enabled=True)

    # ==========================================================================
    # RESUME: find latest periodic epoch checkpoint for this dataset
    # ==========================================================================
    ckpt_path, epoch_completed = find_latest_epoch_ckpt(log_dir, dataset_name)

    if ckpt_path is not None:
        print(f"\nResuming from checkpoint: {ckpt_path}")
        print(f"  Completed epoch: {epoch_completed}  |  "
              f"Remaining epochs: {cfg.epochs - epoch_completed - 1}\n")
        ckpt = torch.load(ckpt_path, map_location="cuda")
        net.load_state_dict(ckpt["net"])
        probe.load_state_dict(ckpt["probe"])
        opt.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["scheduler"])
        # scaler.load_state_dict(ckpt["scaler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        best_OA       = float(ckpt.get("best_OA", ckpt.get("OA", 0.0)))
        start_epoch   = epoch_completed + 1
        run_name      = ckpt.get("run_name", f"lejepa_{time.strftime('%Y%m%d_%H%M%S')}")
    else:
        print("\nNo checkpoint found — starting fresh.\n")
        best_OA     = 0.0
        start_epoch = 0
        run_name    = f"lejepa_{time.strftime('%Y%m%d_%H%M%S')}"

    # ==========================================================================

    wandb.init(project="LeJEPA", name=run_name, config=dict(cfg),
               mode="offline", resume="allow")

    OA = AA = kappa = 0.0
    per_class_acc = np.zeros(train_ds.classes)

    if start_epoch >= cfg.epochs:
        print(f"Already completed {cfg.epochs} epochs. Nothing to do.")
        wandb.finish()
        return

    for epoch in range(start_epoch, cfg.epochs):
        net.train(), probe.train()
        for vs, y in tqdm.tqdm(train, total=len(train), desc=f"[{dataset_name}] ep{epoch}"):
            vs = vs.to("cuda", non_blocking=True)
            y  = y.to("cuda",  non_blocking=True)

            emb, proj   = net(vs)
            inv_loss    = (proj.mean(0, keepdim=True) - proj).square().mean()
            sigreg_loss = sigreg(proj)
            lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
            probe_loss  = F.cross_entropy(probe(emb.detach()), y.repeat_interleave(cfg.V))
            loss        = lejepa_loss + probe_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            wandb.log({
                "train/probe":     probe_loss.item(),
                "train/lejepa":    lejepa_loss.item(),
                "train/sigreg":    sigreg_loss.item(),
                "train/inv":       inv_loss.item(),
                "train/gate_mean": torch.sigmoid(net.branch_gate).mean().item(),
            })

        # ---- evaluation --------------------------------------------------
        net.eval(), probe.eval()
        all_preds, all_labels = [], []

        with torch.inference_mode():
            for vs, y in test:
                vs = vs.to("cuda", non_blocking=True)
                y  = y.to("cuda",  non_blocking=True)
                all_preds.append(probe(net(vs)[0]).argmax(1).cpu().numpy())
                all_labels.append(y.cpu().numpy())

        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        OA, AA, kappa, per_class_acc, _ = compute_metrics(
            all_preds, all_labels, train_ds.classes
        )

        g_mean = torch.sigmoid(net.branch_gate).mean().item()
        wandb.log({
            "test/OA": OA, "test/AA": AA, "test/kappa": kappa,
            "test/epoch": epoch, "gate/spatial_mean": g_mean,
        })

        print(
            f"  [{dataset_name}] ep {epoch:3d} | "
            f"OA {OA*100:.2f}%  AA {AA*100:.2f}%  κ {kappa:.4f}  gate {g_mean:.3f}"
        )

        # shared checkpoint payload
        ckpt_payload = {
            "epoch":         epoch,
            "dataset":       dataset_name,
            "run_name":      run_name,
            "OA":            OA,
            "AA":            AA,
            "kappa":         kappa,
            "best_OA":       best_OA,
            "per_class_acc": per_class_acc.tolist(),
            "branch_gate":   net.branch_gate.data.cpu(),
            "net":           net.state_dict(),
            "probe":         probe.state_dict(),
            "opt":           opt.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "scaler":        scaler.state_dict(),
            "cfg":           dict(cfg),
        }

        # best checkpoint (not used for resume — only for inference)
        if OA > best_OA:
            best_OA = OA
            ckpt_payload["best_OA"] = best_OA
            best_path = os.path.join(log_dir, f"{dataset_name}_{run_name}_best.pt")
            torch.save(ckpt_payload, best_path)
            print(f"    --> best checkpoint  (OA {OA*100:.2f}%)")

        # periodic epoch checkpoint — used for resume
        # saves every 100 epochs AND overwrites the previous one so only
        # the latest periodic checkpoint exists, saving disk space
        if (epoch + 1) % 100 == 0:
            ep_path = os.path.join(log_dir, f"{dataset_name}_{run_name}_ep{epoch+1}.pt")
            torch.save(ckpt_payload, ep_path)
            print(f"    --> epoch checkpoint saved: {ep_path}")

    # ---- final results ---------------------------------------------------
    save_results(results_path, dataset_name, run_name, {
        "epoch":         int(cfg.epochs - 1),
        "OA":            round(OA,      6),
        "AA":            round(AA,      6),
        "kappa":         round(kappa,   6),
        "best_OA":       round(best_OA, 6),
        "per_class_acc": per_class_acc.tolist(),
        "gate_mean":     round(torch.sigmoid(net.branch_gate).mean().item(), 4),
        "cfg":           dict(cfg),
    })

    print(f"\nResults → {results_path}  (key: '{dataset_name}')")
    wandb.finish()


if __name__ == "__main__":
    main()
