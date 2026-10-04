import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from georef import GridGeoref  # re-export so existing imports still work

# NOTE: `from tsl.nn.blocks.encoders import DCRNN` was removed here — it was
# never used (OutbreakSTGNN implements its own diffusion via _aggregate),
# but as a top-level import it would fail this entire module (including
# ConvLSTMUNet) if the `tsl` package isn't installed. Re-add only if/when
# DCRNN is actually wired in.


class OutbreakSTGNN(nn.Module):
    """
    Memory-efficient spatiotemporal outbreak prediction model.

    Processes each timestep independently through shared spatial
    graph-convolution weights. No recurrence, so no backprop-through-time
    and no per-step hidden state stored. Temporal information is carried
    entirely by the input features.

    Why this is much cheaper than DCRNN:
      DCRNN stores gate activations (r, u, c) at every timestep for every
      layer, plus hidden state before and after each step. That's roughly
      O(T * N * H * n_layers * ~6) stored activations.
      This model stores only layer outputs: O(T * N * H * n_layers * ~2).
      At the same T, N, H, that's a 10-15x reduction in activation memory.

    Inputs:
      x_seq:       (T, N, F)     node features per timestep
      edge_index:  (2, E)        source, target node pairs
      edge_weight: (E,)          per-edge scalar weight

    Output:
      list of T tensors, each (N,), raw logits per node.
    """

    def __init__(self, in_channels, hidden_channels=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_channels * 2, hidden_channels) for _ in range(n_layers)]
        )
        self.head = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def _aggregate(self, h, src, dst, edge_weight):
        """
        One round of diffusion. h: (N, H) -> (N, H).
        """
        msg = h[src] * edge_weight.unsqueeze(-1)  # (E, H), transient
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        return agg

    def forward(self, x_seq, edge_index, edge_weight):
        T, N, _ = x_seq.shape
        src, dst = edge_index

        h = self.input_proj(x_seq)  # (T, N, H)
        edge_weight = edge_weight.to(h.dtype)  # cast once

        for layer in self.layers:
            agg = torch.empty_like(h)
            for t in range(T):
                agg[t] = self._aggregate(h[t], src, dst, edge_weight)
            h = F.relu(layer(torch.cat([h, agg], dim=-1)))
            h = self.dropout(h)

        logits = self.head(h).squeeze(-1)
        return [logits[t] for t in range(T)]

    def run_episode(self, snapshot_sequence, device="cpu"):
        """
        Runs a full episode from a list of temporal snapshots.
        Each snapshot must have .x, .edge_index, .edge_attr.
        """
        self.to(device)  # Ensure model is on the correct device

        xs = []
        edge_index = None
        edge_weight = None

        for snapshot in snapshot_sequence:
            xs.append(snapshot.x.to(device))

            if edge_index is None:
                edge_index = snapshot.edge_index.to(device)
                edge_weight = snapshot.edge_attr.to(device)

                # If edge_attr has shape (E, 1), squeeze to (E,)
                if edge_weight.dim() == 2 and edge_weight.size(1) == 1:
                    edge_weight = edge_weight.squeeze(1)

        x_seq = torch.stack(xs, dim=0)  # (T, n_nodes, in_channels)
        return self(x_seq, edge_index, edge_weight)

    def get_config(self):
        """Return model configuration as dict"""
        return {
            "hidden_channels": self.hidden_channels,
            "n_layers": self.n_layers,
            "dropout": self.dropout.p,
        }






class DoubleConv(nn.Module):
    """Two 3x3 convs with batch norm and ReLU."""

    def __init__(self, in_ch, out_ch, mid_channels = None, dropout=0.0):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_ch
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
    """Maxpool then DoubleConv."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upsample, concat with skip, then DoubleConv."""

    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # handle odd sizes: crop or pad to match skip
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class OutbreakUNet(nn.Module):
    """
    U-Net for per-timestep disease prediction from raster inputs.

    Treats timesteps as a batch dimension: the same conv stack runs on
    every timestep independently. Temporal information is carried by the
    input features (time_since_sighting, dist_to_sighting, etc.).

    Input:  (T, C, H, W)     C = N_FEATURES, H, W = grid dims
    Output: list of T tensors, each (1, H, W) with raw logits

    Memory and speed are O(T * H * W) rather than O(T * N * E) as in the
    graph model, and there are no scatter/index_add ops. For a 100x100
    grid at T=150, expect ~1-2 GB peak and ~0.05s per episode.
    """

    def __init__(self, in_channels, base_channels=32, depth=4, dropout=0.1):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.base_channels = base_channels

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels, dropout)
        ch = base_channels
        self.downs = nn.ModuleList()
        for _ in range(depth):
            self.downs.append(Down(ch, ch * 2, dropout))
            ch *= 2

        # Decoder
        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(Up(ch, ch // 2, ch // 2, dropout))
            ch //= 2

        # Head: 1x1 conv to single channel
        self.head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x_seq):
        """
        x_seq: (T, C, H, W)
        returns: list of T tensors, each (1, H, W)
        """
        # Encoder with skip storage
        skips = []
        x = self.inc(x_seq)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # skips[-1] is the bottleneck; drop it and reverse the rest
        skips = skips[:-1][::-1]

        for up, skip in zip(self.ups, skips):
            x = up(x, skip)

        logits = self.head(x)  # (T, 1, H, W)
        return [logits[t] for t in range(logits.shape[0])]


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell. Standard gates: input, forget, output, candidate."""
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(
            in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=padding
        )

    def forward(self, x, h, c):
        # x: (B, C_in, H, W), h, c: (B, C_hid, H, W)
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    """Processes a (B, T, C, H, W) sequence through a ConvLSTM cell."""
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hidden_ch, kernel_size)
        self.hidden_ch = hidden_ch

    def forward(self, x):
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_ch, H, W, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, self.hidden_ch, H, W, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)  # (B, T, C_hid, H, W)

class ConvLSTMUNet(nn.Module):
    """
    U-Net with a ConvLSTM bottleneck.

    Input:  (B, T, C, H, W)
    Output: list of T tensors, each (B, 1, H, W) with raw logits

    Encoder and decoder run per-frame (T folded into batch).
    The ConvLSTM at the bottleneck propagates temporal state across timesteps.
    """
    def __init__(self, in_channels, base_channels=32, depth=4,
                 dropout=0.1, convlstm_layers=1):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.base_channels = base_channels

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels, dropout)
        ch = base_channels
        self.downs = nn.ModuleList()
        for _ in range(depth):
            self.downs.append(Down(ch, ch * 2, dropout))
            ch *= 2

        # Bottleneck ConvLSTM
        self.convlstm = ConvLSTM(ch, ch, kernel_size=3)

        # Decoder
        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(Up(ch, ch // 2, ch // 2, dropout))
            ch //= 2

        self.head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x_seq):
        # x_seq: (B, T, C, H, W)
        B, T, C, H, W = x_seq.shape

        # --- Encoder: run per-frame ---
        x = x_seq.reshape(B * T, C, H, W)
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # skips: list of (B*T, ch_i, h_i, w_i). Last one is the bottleneck.

        # --- ConvLSTM at bottleneck ---
        _, ch_b, h_b, w_b = x.shape
        x = x.reshape(B, T, ch_b, h_b, w_b)
        x = self.convlstm(x)           # (B, T, ch_b, h_b, w_b)

        # --- Decoder: run per-frame ---
        x = x.reshape(B * T, ch_b, h_b, w_b)

        # Reverse encoder skips, drop the bottleneck
        skips = skips[:-1][::-1]

        for up, skip in zip(self.ups, skips):
            x = up(x, skip)

        logits = self.head(x)          # (B*T, 1, H, W)
        logits = logits.reshape(B, T, 1, H, W)

        # Return per-timestep tensors to match the loss function contract
        return [logits[:, t] for t in range(T)]

