"""
unet_report.py

Plain U-Net for per-timestep outbreak forecasting from a 2-channel input:

  channel 0: NDVI                        (static, broadcast across time)
  channel 1: sliding-window report sum   (computed from raw reports)

Forecasting setup
-----------------
  Input  at day t:  ndvi, window[t]
  Target at day t:  ground_truth[t + horizon]
  `horizon=1` predicts tomorrow; `horizon=k` predicts k days ahead.

Data format
-----------
Files written by simModel.save_window contain exactly:
  ndvi         : (H, W)         float32
  reports      : (D, H, W)      uint8
  ground_truth : (D, H, W)      uint8

Window size is a training-time hyperparameter. The sliding window is
computed from `reports` on the fly, so the same data can be retrained
with any window size without regeneration.

Self-contained. No dependency on data.py, model.py, or simModel.py.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# U-Net blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ReportUNet(nn.Module):
    """
    U-Net with 2 input channels (NDVI, sliding-window reports).

    Input:  (D, 2, H, W)
    Output: list of D tensors, each (n_horizons, H, W)

    n_horizons = 1 reduces to single-day prediction.
    """
    def __init__(self, in_channels=2, base_channels=32, depth=4, dropout=0.1,
                 n_horizons=1):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth
        self.n_horizons = n_horizons

        self.inc = DoubleConv(in_channels, base_channels, dropout)
        ch = base_channels
        self.downs = nn.ModuleList()
        for _ in range(depth):
            self.downs.append(Down(ch, ch * 2, dropout))
            ch *= 2

        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(Up(ch, ch // 2, ch // 2, dropout))
            ch //= 2

        # head outputs n_horizons channels: channel c predicts t + c + 1
        self.head = nn.Conv2d(base_channels, n_horizons, 1)

    def forward(self, x):
        # x: (D, 2, H, W)
        skips = []
        h = self.inc(x)
        skips.append(h)
        for down in self.downs:
            h = down(h)
            skips.append(h)

        skips = skips[:-1][::-1]

        for up, skip in zip(self.ups, skips):
            h = up(h, skip)

        logits = self.head(h)  # (D, n_horizons, H, W)
        return [logits[d] for d in range(logits.shape[0])]


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

def build_inputs(ndvi, window, log1p_window=True):
    """
    Build the (D, 2, H, W) input tensor from NDVI and a window array.

    inputs
    --------
    ndvi         : (H, W) or (D, H, W) float array
    window       : (D, H, W) int or float array
    log1p_window : if True, apply log1p to the report counts

    returns
    -------
    x : torch.FloatTensor (D, 2, H, W)
    """
    ndvi = np.asarray(ndvi, dtype=np.float32)
    window = np.asarray(window, dtype=np.float32)

    if window.ndim != 3:
        raise ValueError(f"window must be (D, H, W), got shape {window.shape}")

    D, H, W = window.shape

    if ndvi.ndim == 2:
        ndvi_b = np.broadcast_to(ndvi[None, :, :], (D, H, W))
    elif ndvi.ndim == 3 and ndvi.shape == (D, H, W):
        ndvi_b = ndvi
    else:
        raise ValueError(
            f"ndvi must be (H, W) or (D, H, W); got {ndvi.shape}, "
            f"window is {window.shape}"
        )

    window_ch = np.log1p(window) if log1p_window else window

    x = np.stack([ndvi_b, window_ch], axis=1).astype(np.float32)
    return torch.from_numpy(x)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def tversky_loss_episode(logits_list, y_seq, alpha=0.8, beta=0.2, smooth=1.0):
    """
    Plain Tversky loss aggregated over the whole episode.

    logits_list : list of (1, H, W) tensors, one per timestep
    y_seq       : (D, 1, H, W) binary target
    """
    all_probs = torch.cat([torch.sigmoid(l).float().view(-1) for l in logits_list])
    all_targets = torch.cat([y.float().view(-1) for y in y_seq])

    tp = (all_probs * all_targets).sum()
    fn = ((1 - all_probs) * all_targets).sum()
    fp = (all_probs * (1 - all_targets)).sum()

    return 1 - (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)


def novelty_tversky(logits_list, y_seq, window_seq,
                    alpha=0.8, beta=0.2, smooth=1.0,
                    known_discount=0.5, discount_fn=True):
    k_expected = y_seq.shape[1]
    k_logits = logits_list[0].shape[0]

    if k_logits != k_expected:
        raise ValueError(
            f"logits have {k_logits} horizons but y_seq has {k_expected}. "
            f"Check that episode_from_npz(horizon=k) and "
            f"ReportUNet(n_horizons=k) match."
        )

    all_probs = torch.cat([torch.sigmoid(l).float().flatten() for l in logits_list])
    all_targets = y_seq.float().flatten()

    if isinstance(window_seq, (list, tuple)):
        ws = []
        for w in window_seq:
            if w.dim() == 3 and w.shape[0] == 1:
                w = w.squeeze(0)
            ws.append((w > 0).float())
        known_dhw = torch.stack(ws, dim=0)              # (D', H, W)
    else:
        w = window_seq
        if w.dim() == 4 and w.shape[1] == 1:
            w = w.squeeze(1)                             # (D', H, W)
        known_dhw = (w > 0).float()

    known = known_dhw.unsqueeze(1).expand(-1, k_expected, -1, -1).reshape(-1)

    tp_w = 1.0 - known_discount * known
    fn_w = tp_w if discount_fn else torch.ones_like(known)

    tp = (tp_w * all_probs * all_targets).sum()
    fn = (fn_w * (1 - all_probs) * all_targets).sum()
    fp = (all_probs * (1 - all_targets)).sum()

    return 1 - (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _fit_grid(arr, target_h, target_w, pad_value=0.0):
    """Center pad/crop the last two dims of arr to (target_h, target_w)."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        leading = ()
    elif arr.ndim == 3:
        h, w = arr.shape[-2:]
        leading = arr.shape[:-2]
    else:
        raise ValueError(f"expected 2D or 3D, got shape {arr.shape}")

    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    pad_top = pad_h // 2
    pad_left = pad_w // 2

    if pad_h or pad_w:
        pad_spec = [(0, 0)] * len(leading) + [
            (pad_top, pad_h - pad_top),
            (pad_left, pad_w - pad_left),
        ]
        arr = np.pad(arr, pad_spec, mode="constant", constant_values=pad_value)

    h2, w2 = arr.shape[-2:]
    crop_top = max(0, (h2 - target_h) // 2)
    crop_left = max(0, (w2 - target_w) // 2)

    if crop_top or crop_left:
        slicer = (slice(None),) * len(leading) + (
            slice(crop_top, crop_top + target_h),
            slice(crop_left, crop_left + target_w),
        )
        arr = arr[slicer]

    return arr


def _window_from_reports(reports, size):
    """
    Sliding-window sum over per-day report grids.
    Matches simModel.savewindow.

    reports : (D, H, W) int array
    size    : window length in days (>= 1)
    """
    reports = np.asarray(reports)
    if reports.ndim != 3:
        raise ValueError(f"reports must be (D, H, W), got {reports.shape}")
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")

    cumsum = np.cumsum(reports.astype(np.int32), axis=0)
    D = cumsum.shape[0]

    if size >= D:
        return cumsum.astype(np.uint16)

    result = np.empty_like(cumsum)
    result[:size] = cumsum[:size]
    result[size:] = cumsum[size:] - cumsum[:-size]
    return result.astype(np.uint16)


def episode_from_npz(path, window_size=5, target_size=(80, 80),
                     log1p_window=True, horizon=5, infected_threshold=0.1,
                     subtract_one=False):
    """
    Load one .npz and build inputs / targets for multi-horizon forecasting.

    Input  at day t: (ndvi, window[t])
    Target at day t: (gt[t+1], gt[t+2], ..., gt[t+horizon])
    The last `horizon` input frames are dropped since they have no target.

    horizon = 1 reproduces the single-horizon behavior.

    returns
    -------
    x_seq : (D - horizon, 2, H, W) float32 tensor
    y_seq : (D - horizon, horizon, H, W) float32 tensor
    """
    d = np.load(path, allow_pickle=False)

    ndvi = d["ndvi"].astype(np.float32)
    has_reports = "reports" in d.files
    has_window = "window" in d.files

    if has_reports:
        window = _window_from_reports(d["reports"], window_size).astype(np.int32)
    elif has_window:
        window = d["window"].astype(np.int32)
    else:
        raise ValueError(f"{path}: neither 'reports' nor 'window' present")

    if "ground_truth" in d.files:
        gt = d["ground_truth"].astype(np.float32)
    else:
        gt = (d["I_hist"].astype(np.float32) > infected_threshold).astype(np.float32)

    target_h, target_w = target_size
    ndvi = _fit_grid(ndvi, target_h, target_w, pad_value=0.0)
    window = _fit_grid(window, target_h, target_w, pad_value=0)
    gt = _fit_grid(gt, target_h, target_w, pad_value=0.0)

    if gt.ndim == 2:
        gt = gt[None, ...]

    if subtract_one:
        mask = window > 0
        window = window.copy()
        window[mask] -= 1

    D = window.shape[0]
    k = int(horizon)
    D_out = D - k
    if D_out <= 0:
        raise ValueError(f"horizon={k} but episode only has D={D} days")

    window_in = window[:D_out]                       # (D_out, H, W)

    # stack targets: for input day i, target is gt[i+1 : i+1+k]
    targets = np.stack(
        [gt[i + 1 : i + 1 + k] for i in range(D_out)],
        axis=0,
    )                                                # (D_out, k, H, W)

    x_seq = build_inputs(ndvi, window_in, log1p_window=log1p_window)
    y_seq = torch.from_numpy(targets.astype(np.float32))

    return x_seq, y_seq


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_unet(
    sim_data_dir,
    window_size=5,
    target_size=(80, 80),
    log1p_window=True,
    horizon=5,
    subtract_one=True,
    known_discount=0.5,
    discount_fn=True,
    horizon_weights=None,
    model=None,
    n_epochs=25,
    lr=1e-3,
    base_channels=32,
    depth=4,
    dropout=0.1,
    device="cuda",
    tversky_alpha=0.8,
    tversky_beta=0.2,
    val_split=0.2,
    shuffle=True,
    seed=42,
    use_amp=True,
    verbose_metrics=True,
    save_path="unet.pth",
    on_epoch_end=None,
):
    """
    Train a ReportUNet with a multi-horizon head on NDVI + sliding-window
    reports.

    Input  at day t:  (ndvi, window[t])
    Target at day t:  gt[t+1 : t+horizon+1]   — k future frames
    Output at day t:  k channels of logits

    inputs
    --------
    horizon         : k, number of future days to predict simultaneously
    horizon_weights : optional sequence of k floats. If provided, each
                      horizon's contribution to the loss is scaled.
                      Try [0.5, 1.0, 1.5, 2.0, 2.5] for k=5 to emphasize
                      the far horizons. Default None = equal weighting.
    verbose_metrics : print per-horizon dice each epoch
    on_epoch_end    : optional callable(epoch, history, model). Called after
                      each epoch completes. Use for MLflow logging, CSV
                      writing, progress bars, early stopping hooks, etc.
                      The function has no MLflow dependency itself.
    (other parameters same as before)

    returns
    -------
    model, history
        history keys:
          train_loss     : list of floats per epoch
          val_loss       : list of floats per epoch
          horizon_dice   : list of k-length lists, one per epoch
    """
    from pathlib import Path

    files = sorted(Path(sim_data_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {sim_data_dir}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    files = [files[i] for i in order]

    n_val = int(len(files) * val_split)
    val_files = files[:n_val]
    train_files = files[n_val:]

    k = int(horizon)
    if horizon_weights is not None:
        hw = torch.tensor(list(horizon_weights), dtype=torch.float32)
        assert len(hw) == k, f"horizon_weights must have length {k}"
    else:
        hw = None

    print(f"Loaded {len(files)} episodes: {len(train_files)} train, "
          f"{len(val_files)} val")
    print(f"horizon={k}  window_size={window_size}  "
          f"target_size={target_size}  subtract_one={subtract_one}")
    print(f"loss: novelty_tversky  known_discount={known_discount}  "
          f"discount_fn={discount_fn}")
    if hw is not None:
        print(f"horizon_weights: {hw.tolist()}")

    # --- model ---
    if model is None:
        model = ReportUNet(
            in_channels=2,
            base_channels=base_channels,
            depth=depth,
            dropout=dropout,
            n_horizons=k,
        ).to(device)
    else:
        model = model.to(device)
        if getattr(model, "n_horizons", 1) != k:
            raise ValueError(
                f"model.n_horizons={model.n_horizons} but horizon={k}"
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    amp_enabled = use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # --- loss ---
    def _loss(logits_list, y_seq, x_seq):
        window_ch = x_seq[:, 1:2, :, :]
        if hw is None:
            return novelty_tversky(
                logits_list, y_seq, window_ch,
                alpha=tversky_alpha, beta=tversky_beta,
                known_discount=known_discount, discount_fn=discount_fn,
            )
        hw_dev = hw.to(y_seq.device)
        total = 0.0
        weight_sum = 0.0
        for c in range(k):
            logits_c = [logits[c] for logits in logits_list]
            y_c = y_seq[:, c:c+1, :, :]
            loss_c = novelty_tversky(
                logits_c, y_c, window_ch,
                alpha=tversky_alpha, beta=tversky_beta,
                known_discount=known_discount, discount_fn=discount_fn,
            )
            total = total + hw_dev[c] * loss_c
            weight_sum = weight_sum + hw_dev[c]
        return total / weight_sum

    def _per_horizon_dice(logits_list, y_seq, threshold=0.5):
        preds = torch.stack(logits_list).sigmoid()
        preds = (preds > threshold).float()
        targets = y_seq.float()
        dice = torch.zeros(k, device=y_seq.device)
        for c in range(k):
            p = preds[:, c]
            t = targets[:, c]
            inter = (p * t).sum()
            denom = p.sum() + t.sum()
            dice[c] = (2.0 * inter / denom) if denom > 0 else 1.0
        return dice.cpu().numpy()

    history = {
        "train_loss": [],
        "val_loss": [],
        "horizon_dice": [],
        "config": {
            "sim_data_dir": str(sim_data_dir),
            "window_size": window_size,
            "target_size": list(target_size),
            "log1p_window": log1p_window,
            "horizon": k,
            "subtract_one": subtract_one,
            "known_discount": known_discount,
            "discount_fn": discount_fn,
            "horizon_weights": None if hw is None else hw.tolist(),
            "n_epochs": n_epochs,
            "lr": lr,
            "base_channels": base_channels,
            "depth": depth,
            "dropout": dropout,
            "device": device,
            "tversky_alpha": tversky_alpha,
            "tversky_beta": tversky_beta,
            "val_split": val_split,
            "seed": seed,
            "use_amp": use_amp,
            "n_train_files": len(train_files),
            "n_val_files": len(val_files),
        },
    }

    for epoch in range(n_epochs):
        # ---------- training ----------
        model.train()
        epoch_files = train_files[:]
        if shuffle:
            rng.shuffle(epoch_files)

        epoch_loss = 0.0
        n_done = 0

        for path in epoch_files:
            optimizer.zero_grad()

            x_seq, y_seq = episode_from_npz(
                str(path),
                window_size=window_size,
                target_size=target_size,
                log1p_window=log1p_window,
                horizon=k,
                subtract_one=subtract_one,
            )
            x_seq = x_seq.to(device)
            y_seq = y_seq.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16,
                                    enabled=amp_enabled):
                logits_list = model(x_seq)
                loss = _loss(logits_list, y_seq, x_seq)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_done += 1

        train_loss = epoch_loss / max(n_done, 1)
        history["train_loss"].append(train_loss)

        # ---------- validation ----------
        val_loss = float("nan")
        dice_accum = np.zeros(k, dtype=np.float64)
        dice_count = 0

        if val_files:
            model.eval()
            v = 0.0
            vn = 0
            with torch.no_grad():
                for path in val_files:
                    x_seq, y_seq = episode_from_npz(
                        str(path),
                        window_size=window_size,
                        target_size=target_size,
                        log1p_window=log1p_window,
                        horizon=k,
                        subtract_one=subtract_one,
                    )
                    x_seq = x_seq.to(device)
                    y_seq = y_seq.to(device)

                    with torch.amp.autocast("cuda", dtype=torch.float16,
                                            enabled=amp_enabled):
                        logits_list = model(x_seq)
                        loss = _loss(logits_list, y_seq, x_seq)

                    v += loss.item()
                    vn += 1
                    dice_accum += _per_horizon_dice(logits_list, y_seq)
                    dice_count += 1

            val_loss = v / max(vn, 1)
            mean_dice = dice_accum / max(dice_count, 1)
            history["val_loss"].append(val_loss)
            history["horizon_dice"].append(mean_dice.tolist())

            if verbose_metrics:
                dice_str = "  ".join(
                    f"+{c+1}={mean_dice[c]:.3f}" for c in range(k)
                )
                print(f"epoch {epoch+1}/{n_epochs}  "
                      f"train={train_loss:.4f}  val={val_loss:.4f}")
                print(f"    dice: {dice_str}")
            else:
                print(f"epoch {epoch+1}/{n_epochs}  "
                      f"train={train_loss:.4f}  val={val_loss:.4f}")
        else:
            history["val_loss"].append(float("nan"))
            print(f"epoch {epoch+1}/{n_epochs}  train={train_loss:.4f}")

        # ---------- callback ----------
        if on_epoch_end is not None:
            on_epoch_end(epoch, history, model)

    # ---------- save ----------
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": 2,
            "base_channels": base_channels,
            "depth": depth,
            "dropout": dropout,
            "n_horizons": k,
        },
        "training_info": {
            "window_size": window_size,
            "target_size": list(target_size),
            "log1p_window": log1p_window,
            "horizon": k,
            "subtract_one": subtract_one,
            "known_discount": known_discount,
            "discount_fn": discount_fn,
            "horizon_weights": None if hw is None else hw.tolist(),
            "tversky_alpha": tversky_alpha,
            "tversky_beta": tversky_beta,
        },
        "history": history,
    }
    torch.save(ckpt, save_path)
    print(f"Model saved to {save_path}")

    return model, history


def load_unet(filepath, device="cpu"):
    """Load a ReportUNet checkpoint saved by train_unet."""
    ckpt = torch.load(filepath, map_location=device)
    cfg = ckpt["model_config"]
    model = ReportUNet(
        in_channels=cfg.get("in_channels", 2),
        base_channels=cfg.get("base_channels", 32),
        depth=cfg.get("depth", 4),
        dropout=cfg.get("dropout", 0.1),
        n_horizons=cfg.get("n_horizons", 1), 
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def _sanity_check_losses():
    """
    Confirm novelty_tversky behaves as expected:
      - plain (known_discount=0) matches tversky_loss_episode
      - increasing known coverage increases loss
    """
    torch.manual_seed(0)
    D, H, W = 3, 8, 8
    logits = [torch.zeros(1, H, W)] * D
    targets = torch.zeros(D, 1, H, W)
    targets[:, :, 0:2, 0:2] = 1.0

    plain = tversky_loss_episode(logits, targets).item()
    zero_disc = novelty_tversky(
        logits, targets, torch.zeros(D, 1, H, W), known_discount=0.0
    ).item()
    assert abs(plain - zero_disc) < 1e-6, (plain, zero_disc)

    window = torch.zeros(D, 1, H, W)
    window[:, :, 0, 0] = 1.0
    novel_25 = novelty_tversky(logits, targets, window, known_discount=0.5).item()

    window_all = torch.ones(D, 1, H, W)
    novel_all = novelty_tversky(logits, targets, window_all, known_discount=0.5).item()

    print(f"plain:          {plain:.4f}")
    print(f"novelty 25%:    {novel_25:.4f}")
    print(f"novelty 100%:   {novel_all:.4f}")
    assert novel_25 > plain
    assert novel_all > novel_25
    print("sanity check passed")


if __name__ == "__main__":
    _sanity_check_losses()