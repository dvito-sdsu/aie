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

    The D axis is folded into the batch dimension. Each timestep is processed
    independently; the model has no memory across time.

    Input:  (D, 2, H, W)
    Output: list of D tensors, each (1, H, W)

    H and W must be divisible by 2**depth. With depth=4, use 112x112 or 128x128.
    """
    def __init__(self, in_channels=2, base_channels=32, depth=4, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth

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

        self.head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x):
        # x: (D, 2, H, W)
        skips = []
        h = self.inc(x)
        skips.append(h)
        for down in self.downs:
            h = down(h)
            skips.append(h)

        skips = skips[:-1][::-1]  # drop bottleneck, reverse

        for up, skip in zip(self.ups, skips):
            h = up(h, skip)

        logits = self.head(h)  # (D, 1, H, W)
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
    """
    Tversky loss with reduced reward for predictions on already-known cells.

    A cell is "known" at timestep t if the input window at t had a nonzero
    value there (post any preprocessing the model received). Known cells
    contribute less to the TP and (optionally) FN terms, so the model is
    rewarded primarily for discoveries at new locations.

    inputs
    --------
    logits_list    : list of (1, H, W) tensors, one per timestep
    y_seq          : (D, 1, H, W) binary target
    window_seq     : (D, 1, H, W) window channel the model saw as input,
                     or a list of (1, H, W) tensors.
    alpha, beta    : Tversky weights on FN and FP
    smooth         : denominator smoothing
    known_discount : 0 = plain Tversky; 0.5 = known cells count half;
                     1 = known cells contribute nothing to TP/FN
    discount_fn    : if True, also reduce FN weight on known cells

    returns
    -------
    loss : scalar tensor
    """
    all_probs = torch.cat([torch.sigmoid(l).float().view(-1) for l in logits_list])
    all_targets = torch.cat([y.float().view(-1) for y in y_seq])

    if isinstance(window_seq, (list, tuple)):
        known = torch.cat([(w > 0).float().view(-1) for w in window_seq])
    else:
        known = (window_seq > 0).float().view(-1)

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


def episode_from_npz(path, window_size=5, target_size=(112, 112),
                     log1p_window=True, horizon=1, infected_threshold=0.1,
                     subtract_one=False):
    """
    Load one .npz and build inputs / targets for forecasting.

    Window source priority:
      1. If the file has 'reports' (always true for save_window output),
         compute the window from reports using `window_size`.
      2. Else if the file has a precomputed 'window', use it as-is.
      3. Else error.

    Input  for day t: (ndvi, window[t])
    Target for day t: ground_truth[t + horizon]
    The last `horizon` input frames are dropped since they have no target.

    inputs
    --------
    path          : path to the .npz
    window_size   : int, sliding window length in days (>= 1)
    target_size   : (H, W) output grid size, divisible by 2**depth
    log1p_window  : log1p transform on the report channel
    horizon       : days ahead to predict (0 = same day, 1 = tomorrow)
    infected_threshold : used only if ground_truth isn't in the file
    subtract_one  : subtract 1 from every nonzero window count before log1p.
                    Zero entries are left alone.

    returns
    -------
    x_seq : (D - horizon, 2, H, W) float32 tensor
    y_seq : (D - horizon, 1, H, W) float32 tensor
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
        raise ValueError(
            f"{path}: neither 'reports' nor 'window' present, cannot "
            f"build the report channel"
        )

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
    if horizon >= D:
        raise ValueError(f"horizon={horizon} but episode only has D={D} days")

    window_in = window[: D - horizon]
    gt_target = gt[horizon: D]

    x_seq = build_inputs(ndvi, window_in, log1p_window=log1p_window)
    y_seq = torch.from_numpy(gt_target[:, None, :, :].astype(np.float32))

    return x_seq, y_seq


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_unet(
    sim_data_dir,
    window_size=5,
    target_size=(112, 112),
    log1p_window=True,
    horizon=1,
    subtract_one=False,
    known_discount=0.5,
    discount_fn=True,
    model=None,
    n_epochs=20,
    lr=1e-3,
    base_channels=32,
    depth=4,
    dropout=0.1,
    device="cpu",
    tversky_alpha=0.8,
    tversky_beta=0.2,
    val_split=0.2,
    shuffle=True,
    seed=42,
    use_amp=True,
    save_path="unet_report.pth",
):
    """
    Train a plain U-Net on NDVI + sliding-window reports to forecast
    infection `horizon` days ahead.

    inputs
    --------
    sim_data_dir   : folder of .npz files from simModel.save_window
    window_size    : sliding window length in days (>= 1)
    target_size    : (H, W) grid size, divisible by 2**depth
    log1p_window   : log1p transform on the report channel
    horizon        : days ahead to predict (0 = same day, 1 = tomorrow)
    subtract_one   : subtract 1 from nonzero window entries before log1p
    known_discount : novelty weight; 0 = plain Tversky, 0.5 = recommended
    discount_fn    : also reduce FN weight on known cells
    model          : pre-existing ReportUNet or None
    n_epochs       : training epochs
    lr             : learning rate
    base_channels  : U-Net base width
    depth          : U-Net depth
    dropout        : U-Net dropout
    device         : "cpu" or "cuda"
    tversky_alpha, tversky_beta : loss weights
    val_split      : fraction held out for validation
    shuffle        : reshuffle training set each epoch
    seed           : RNG seed
    use_amp        : fp16 autocast on CUDA
    save_path      : where to write the final checkpoint

    returns
    -------
    model, history
    """
    files = sorted(Path(sim_data_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {sim_data_dir}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    files = [files[i] for i in order]

    n_val = int(len(files) * val_split)
    val_files = files[:n_val]
    train_files = files[n_val:]

    # startup diagnostic
    with np.load(files[0], allow_pickle=False) as d:
        keys = sorted(d.files)
    print(f"Loaded {len(files)} episodes: {len(train_files)} train, {len(val_files)} val")
    print(f"File keys: {keys}")
    print(f"window_size: {window_size}   horizon: {horizon}   "
          f"target_size: {target_size}   subtract_one: {subtract_one}")
    print(f"Loss: novelty_tversky   known_discount={known_discount}   "
          f"discount_fn={discount_fn}")

    if model is None:
        model = ReportUNet(
            in_channels=2, base_channels=base_channels,
            depth=depth, dropout=dropout,
        ).to(device)
    else:
        model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    amp_enabled = use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    def _loss(logits_list, y_seq, x_seq):
        window_ch = x_seq[:, 1:2, :, :]
        return novelty_tversky(
            logits_list, y_seq, window_ch,
            alpha=tversky_alpha, beta=tversky_beta,
            known_discount=known_discount,
            discount_fn=discount_fn,
        )

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        model.train()
        epoch_files = train_files[:]
        if shuffle:
            rng.shuffle(epoch_files)

        epoch_loss = 0.0
        n_done = 0

        for path in epoch_files:
            optimizer.zero_grad()

            x_seq, y_seq = episode_from_npz(
                path,
                window_size=window_size,
                target_size=target_size,
                log1p_window=log1p_window,
                horizon=horizon,
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

        val_loss = float("nan")
        if val_files:
            model.eval()
            v = 0.0
            vn = 0
            with torch.no_grad():
                for path in val_files:
                    x_seq, y_seq = episode_from_npz(
                        path,
                        window_size=window_size,
                        target_size=target_size,
                        log1p_window=log1p_window,
                        horizon=horizon,
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
            val_loss = v / max(vn, 1)
            history["val_loss"].append(val_loss)
            print(f"epoch {epoch+1}/{n_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")
        else:
            history["val_loss"].append(float("nan"))
            print(f"epoch {epoch+1}/{n_epochs}  train={train_loss:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": 2,
            "base_channels": base_channels,
            "depth": depth,
            "dropout": dropout,
        },
        "training_info": {
            "window_size": window_size,
            "target_size": list(target_size),
            "log1p_window": log1p_window,
            "horizon": horizon,
            "subtract_one": subtract_one,
            "known_discount": known_discount,
            "discount_fn": discount_fn,
            "tversky_alpha": tversky_alpha,
            "tversky_beta": tversky_beta,
        },
    }, save_path)
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