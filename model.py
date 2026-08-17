import torch
import torch.nn as nn
import torch.nn.functional as F
from tsl.nn.blocks.encoders import DCRNN
import numpy as np


class OutbreakSTGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, K=2, dropout=0.2):
        """
        in_channels: number of node features per timestep
        hidden_channels: size of the recurrent hidden state per node
        K: diffusion-convolution filter size (tsl calls this `k`)
        """
        super().__init__()
        self.recurrent = DCRNN(
            input_size=in_channels,
            hidden_size=hidden_channels,
            n_layers=1,
            k=K,
            return_only_last_state=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x_seq, edge_index, edge_weight):
        """
        x_seq: (T, n_nodes, in_channels)
        edge_index: (2, E)
        edge_weight: (E,) or (E, 1)
        returns: list of tensors, one per timestep, each of shape (n_nodes,)
        """
        # tsl wants [batch, time, nodes, features]
        x = x_seq.unsqueeze(0)  # (1, T, n_nodes, in_channels)

        out = self.recurrent(x, edge_index, edge_weight)

        # Handle possible return types from tsl's DCRNN:
        if isinstance(out, tuple):  # (output, hidden_final) likely shape
            h_seq = out[0]  # (output, h_final)
        elif isinstance(out, list):
            h_seq = out[-1]  # last layer output
        else:  # tensor
            h_seq = out

        # Expected shape: (1, T, n_nodes, hidden_channels)
        h_seq = h_seq.squeeze(0)  # (T, n_nodes, hidden_channels)
        out = self.dropout(F.relu(h_seq))
        logits = self.head(out).squeeze(-1)  # (T, n_nodes)

        # Keep original behaviour: return list of per‑timestep predictions
        return [logits[t] for t in range(logits.shape[0])]

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


class GridGeoref:
    """
    maps between geographic coordinates (lat, lon) and grid indices (row, col)

    The grid is assumed to be perfectly rectangular in physical space,
    with constant cell spacing (dist_per_cell) in meters along both axes.
    The mapping uses a spherical Earth approximation with radius R.

    The longitude scale is fixed at the centre latitude; this is accurate
    only for grids with limited latitudinal extent (see class docstring).
    """

    R = 6378137.0  # earth's equatorial radius in meters
    RAD_PER_DEG = np.pi / 180.0
    DEG_PER_RAD = 180.0 / np.pi

    def __init__(self, center_coords, dist_per_cell, n_rows, n_cols, center_idx=None):
        """
        inputs
        -------
            center_coords: (lat, lon) of the reference cell.
            dist_per_cell: grid spacing in meters (must be > 0).
            n_rows, n_cols: grid dimensions (must be > 0).
            center_idx: (row, col) of the reference cell, if none,
                        the cell closest to the array's geometric centre is used.
        """
        # --- Input validation ---
        if dist_per_cell <= 0:
            raise ValueError("dist_per_cell must be positive")
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("n_rows and n_cols must be positive integers")
        center_lat, center_lon = center_coords
        if not (-90.0 <= center_lat <= 90.0) or not (-180.0 <= center_lon <= 180.0):
            raise ValueError("Invalid latitude or longitude")
        if abs(center_lat) >= 89.9:  # avoid cos(lat) ~ 0
            raise ValueError(
                "Centre latitude too close to a pole; cosine would be near zero"
            )

        if center_idx is None:
            center_idx = (n_rows // 2, n_cols // 2)
        self.center_row, self.center_col = center_idx
        self.center_lat, self.center_lon = center_coords
        self.dist_per_cell = dist_per_cell
        self.n_rows, self.n_cols = n_rows, n_cols

        # Precompute scales (degrees per cell) for fast conversion.
        # lat_scale is constant because meridians are parallel on a sphere.
        # lon_scale is fixed at the centre latitude – see accuracy note above.
        cos_lat = np.cos(self.center_lat * self.RAD_PER_DEG)
        self.lat_scale = (dist_per_cell / self.R) * self.DEG_PER_RAD
        self.lon_scale = (dist_per_cell / (self.R * cos_lat)) * self.DEG_PER_RAD

    @classmethod
    def from_locs(cls, locs, dist_per_cell, center_idx=None):
        """
        Alternative constructor that extracts centre coordinates from a
        pre‑existing lat/lon array (as generated by generate_latlon_matrix).

        Assumes the array is exactly consistent with dist_per_cell and the
        rectangular grid geometry; no validation of that consistency is done.
        """
        n_rows, n_cols = locs.shape[0], locs.shape[1]
        if center_idx is None:
            center_idx = (n_rows // 2, n_cols // 2)
        center_lat = locs[center_idx[0], center_idx[1], 0]
        center_lon = locs[center_idx[0], center_idx[1], 1]
        return cls(
            center_coords=(center_lat, center_lon),
            dist_per_cell=dist_per_cell,
            n_rows=n_rows,
            n_cols=n_cols,
            center_idx=center_idx,
        )

    def rc_to_latlon(self, row, col):
        """
        Convert grid indices (row, col) to geographic coordinates (lat, lon).

        Accepts floating point indices, allowing interpolation of positions
        between cell centres. The result is the exact lat/lon of that point
        under the spherical approximation.
        """
        lat = self.center_lat + (self.center_row - row) * self.lat_scale
        lon = self.center_lon + (col - self.center_col) * self.lon_scale
        return lat, lon

    def latlon_to_rc(self, lat, lon):
        """
        Convert geographic coordinates (lat, lon) to the nearest valid grid cell.

        The conversion is performed by inverting the forward formula, then
        rounding to the closest integer index. The result is clipped to
        [0, n_rows-1] and [0, n_cols-1] to guarantee a valid index.
        """
        row_float = self.center_row - (lat - self.center_lat) / self.lat_scale
        col_float = self.center_col + (lon - self.center_lon) / self.lon_scale
        row = round(row_float)
        col = round(col_float)
        # Clip to valid range; rounding then clipping is stable and safe.
        row = max(0, min(row, self.n_rows - 1))
        col = max(0, min(col, self.n_cols - 1))
        return row, col

    def node_id(self, row, col):
        """
        Return a linear index for the cell at (row, col)
        """
        return row * self.n_cols + col

    def rc_of_node(self, node_id):
        """
        Recover (row, col) from a linear node_id
        """
        return divmod(node_id, self.n_cols)
