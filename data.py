import numpy as np
import torch
from pathlib import Path
from PIL import Image
import os
import random
from skimage import measure
from shapely.geometry import Polygon, MultiPoint
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from model2 import GridGeoref


def genReports(
    Sus,
    Infected,
    day,
    locs,
    dist_per_grid,
    seed=None,
    fp=0.01,
    fn=0.01,
    plantspergrid=1,
):
    """
    goal
    change simulation data into realistic reports of format [lat, lon, day]

    INPUTS
    --------
    sus = susceptible matrix of size MxN floats in range [0,1]
    infected = infected matrix of size MxN floats in range [0,1]

    locs = 3d array from generate_latlon_matrix, shape (M, N, 2) -> [...,0]=lat, [...,1]=lon
    dist_per_grid = size of one grid cell, in the SAME units as locs (degrees lat/lon).
        Used to jitter a report's exact position to somewhere inside its grid cell instead of always the cell center.

    fp = false positive rate for reports (per healthy plant, per grid cell)
    fn = false negative rate for reports (per truly-infected plant that got
         detected by the Poisson draw below)

    RETURNS
    ---------
    Nx3 ndarray of reported disease detections, columns = [lat, lon, day]
    """
    rng = np.random.default_rng(seed)

    # true positives
    # expected number of infected plants "found" per cell
    counts = rng.poisson(Infected * plantspergrid)
    rows, cols = np.nonzero(counts)
    n = counts[rows, cols]
    rows_rep = np.repeat(rows, n)
    cols_rep = np.repeat(cols, n)

    # false negatives
    # fn chance of being missed by the surveyor/sensor
    if len(rows_rep) > 0:
        keep = rng.random(len(rows_rep)) >= fn
        rows_rep = rows_rep[keep]
        cols_rep = cols_rep[keep]

    # adding false positives
    fp_counts = rng.poisson(Sus * plantspergrid * fp)
    fp_rows, fp_cols = np.nonzero(fp_counts)
    fp_n = fp_counts[fp_rows, fp_cols]
    fp_rows_rep = np.repeat(fp_rows, fp_n)
    fp_cols_rep = np.repeat(fp_cols, fp_n)

    # combine true + false positives
    all_rows = np.concatenate([rows_rep, fp_rows_rep])
    all_cols = np.concatenate([cols_rep, fp_cols_rep])

    if len(all_rows) == 0:
        return np.empty((0, 3), dtype=float)

    # shuffle so true-positive and false-positive reports are interleaved
    # rather than false positives always trailing at the end
    perm = rng.permutation(len(all_rows))
    all_rows = all_rows[perm]
    all_cols = all_cols[perm]

    # locs is (M, N, 2) with the last axis ordered [lat, lon]
    lat = locs[all_rows, all_cols, 0].astype(float)
    lon = locs[all_rows, all_cols, 1].astype(float)

    # jitter each report to a random point inside its grid cell rather than
    # always reporting the exact cell-center coordinate
    lat = lat + rng.uniform(-dist_per_grid / 2, dist_per_grid / 2, size=lat.shape)
    lon = lon + rng.uniform(-dist_per_grid / 2, dist_per_grid / 2, size=lon.shape)

    day_col = np.full(lat.shape, day, dtype=float)

    reports = np.column_stack([lat, lon, day_col])
    # print(reports.shape)
    return reports


def reports_to_sightings(reports):
    # small function to convert the output from genReports() to a format readable by the st-gnn
    # can be implemented into genReports later
    return [{"lat": r[0], "lon": r[1], "t": r[2]} for r in reports]


def generate_latlon_matrix(input_matrix, center_coords, dist_per_cell=15):
    """
    creates a matrix of shape input_matrix with each value relating of the estimated long/lat

    INPUTS
    ---------
    input_matrix = 2d numpy array
    center_idx = tuple (row,column) indicating the indicie of the center (might be removable)
    center_coords = tuple (lat, long) cooresponding to the coord of the center idx
    dist_per_cell = distance for each cell, in meters

    RETURNS
    ---------
    numpy array with shape of input and one additional layer, representing each spaces relative coordinate
    """
    rows, cols = input_matrix.shape
    center_row, center_col = rows // 2, cols // 2
    center_lat, center_lon = center_coords

    # constants
    R = 6378137.0  # radius of the earth
    RAD_PER_DEG = np.pi / 180.0
    DEG_PER_RAD = 180.0 / np.pi

    cos_lat = np.cos(center_lat * RAD_PER_DEG)
    lat_scale = (dist_per_cell / R) * DEG_PER_RAD
    lon_scale = (dist_per_cell / (R * cos_lat)) * DEG_PER_RAD

    # creating meshgrid
    y_indices = -(np.arange(rows) - center_row)
    x_indices = np.arange(cols) - center_col

    x_offsets, y_offsets = np.meshgrid(x_indices, y_indices)

    # calculating
    target_lats = center_lat + (y_offsets * lat_scale)
    target_lons = center_lon + (x_offsets * lon_scale)

    # combining
    coordinate_matrix = np.dstack((target_lats, target_lons))

    return coordinate_matrix


def build_connections(n_rows, n_cols, connections=8):
    # allows for the nn to connect nearby nodes to each other, generates edges and weights
    if connections not in [4, 8]:
        raise ValueError(f"Connections must be 4 or 8, passed: {connections}")
    node_ids = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)

    offsets_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    offsets_8_extra = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    offsets = offsets_4 + (offsets_8_extra if connections == 8 else [])

    src_chunks = []
    dst_chunks = []
    weight_chunks = []

    for dr, dc in offsets:
        r_src_start, r_src_end = max(0, -dr), n_rows - max(0, dr)
        c_src_start, c_src_end = max(0, -dc), n_cols - max(0, dc)

        src_block = node_ids[r_src_start:r_src_end, c_src_start:c_src_end]
        dst_block = node_ids[
            r_src_start + dr : r_src_end + dr, c_src_start + dc : c_src_end + dc
        ]

        src_chunks.append(src_block.reshape(-1))
        dst_chunks.append(dst_block.reshape(-1))

        w = 1.0 if (dr == 0 or dc == 0) else 1.0 / np.sqrt(2)
        weight_chunks.append(np.full(src_block.size, w, dtype=np.float32))

    src = np.concatenate(src_chunks)
    dst = np.concatenate(dst_chunks)
    edge_weight_np = np.concatenate(weight_chunks)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(edge_weight_np, dtype=torch.float)
    return edge_index, edge_weight


def load_random_ndvi(foldername, used=None):
    """
    function to load a random ndvi file
    expects every file in the folder provided to be a .png
    expects filenames in the format 'lat-lon.png'

    inputs

    foldername = the path to the folder containing the images
    used = previously loaded (lat, lon) pairs as an Nx2 numpy array,
    can be left empty if replacement is fine

    returns

    ndvi = the loaded ndvi data in data type float
    used = Nx2 numpy array of [lat, lon] pairs, with the new entry added at the top
    """
    foldername = "./ndvidata/" + foldername
    if not os.path.isdir(foldername):
        raise NotADirectoryError(f"Folder not found: {foldername}")

    if used is None or (isinstance(used, np.ndarray) and used.size == 0):
        used = np.empty((0, 2))
    elif isinstance(used, list):
        used = np.array(used)
        if used.size == 0:
            used = np.empty((0, 2))

    # get all png files in the folder

    all_files = [f for f in os.listdir(foldername) if f.lower().endswith(".png")]

    if not all_files:
        raise FileNotFoundError(f"No .png files found in {foldername}")

    # exclude already-used files
    available = [f for f in all_files if f not in used]

    if not available:
        raise ValueError(f"All .png files in {foldername} have already been used")

    # pick a random file from the remaining ones
    chosen_file = random.choice(available)
    filepath = os.path.join(foldername, chosen_file)
    lat, lon = parse_latlon(chosen_file)

    # load as 16-bit grayscale, convert to float
    img = Image.open(filepath)
    ndvi = (
        np.array(img).astype(float) / 65536
    )  # 16 bit data conversion back to range [0,1]

    # add new entry at the top of used
    new_entry = np.array([[lat, lon]])
    used = np.vstack([new_entry, used])

    return ndvi, used


def parse_latlon(fname):
    """
    parses a filename in the format lat-lon.png into a (lat, lon) tuple

    inputs

    fname = file name

    returns

    tuple with format (lat,lon) as floats
    """
    name = os.path.splitext(fname)[0]
    lat_str, lon_str = name.split("_")
    return float(lat_str), float(lon_str)


def _rc_to_latlon_interp(row, col, georef):
    """
    helper function for create_polygon
    helps for fractional row/col values

    inputs

    row = row position (float) to be calculated with the geo ref
    col = same as above but for column
    georef = GridGeoref instance (see model.py)

    returns

    lat
    lon
    """
    return georef.rc_to_latlon(row, col)


def create_polygon(
    prob_grid,
    georef,
    threshold=0.5,
    method="contour",
    simplify_tolerance_cells=0.5,
):
    """
    prob_grid: (n_rows, n_cols) probabilities in [0, 1]
    georef: the GridGeoref used to build the graph, for coordinate conversion
    threshold: probability cutoff for "infected"
    method: "contour" or "convex_hull"
    simplify_tolerance_cells: shapely simplify tolerance, in grid-cell units
        (only used for method="contour"; smooths jagged pixel edges)

    Returns: list of (lat, lon) tuples describing the polygon boundary, or
    None if nothing is above threshold.
    """
    if method == "convex_hull":
        rows, cols = np.nonzero(prob_grid >= threshold)
        if len(rows) == 0:
            return None
        points = [georef.rc_to_latlon(r, c) for r, c in zip(rows, cols)]
        hull = MultiPoint([(lon, lat) for lat, lon in points]).convex_hull
        if hull.geom_type != "Polygon":
            return None
        return [(lat, lon) for lon, lat in hull.exterior.coords]

    elif method == "contour":
        contours = measure.find_contours(prob_grid, level=threshold)
        if not contours:
            return None
        # keep the largest contour by enclosed area (in cell units)
        largest = max(
            contours, key=lambda c: Polygon(c[:, ::-1]).area if len(c) >= 4 else 0
        )
        poly = Polygon(
            largest[:, ::-1]
        )  # find_contours gives (row, col); Polygon wants (x, y) = (col, row)
        if not poly.is_valid or poly.is_empty:
            return None
        poly = poly.simplify(simplify_tolerance_cells, preserve_topology=True)
        coords = list(poly.exterior.coords)
        return [georef.rc_to_latlon(r, c) for c, r in coords]

    else:
        raise ValueError(f"unknown method: {method}")


def plot_overlay(
    grid, polygon, georef, title=None, cmap="viridis", ax=None, save_path=None
):
    """
    visually displays the polygon around the infected area


    inputs
    ---------
    grid = the area that will be contained (infected matrix)

    polygon - output from create polygon

    georef - needed to convert the polygon fractional locations to index values

    title, cmap, ax = options to change the matplotlib resultant figure

    save_path = to save the generated image

    returns
    --------
    the matplot figure
    """

    # creating the base figure
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    im = ax.imshow(grid, cmap=cmap, origin="upper")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if polygon is not None and len(polygon) > 0:  # making sure the polygon is real
        # conversion to int indicies
        rows_poly, cols_poly = [], []
        for lat, lon in polygon:  # slow but only used to visualize for people
            row = georef.center_row - (lat - georef.center_lat) / georef.lat_scale
            col = georef.center_col + (lon - georef.center_lon) / georef.lon_scale
            rows_poly.append(row)
            cols_poly.append(col)
        # close the loop back to the first vertex
        rows_poly.append(rows_poly[0])
        cols_poly.append(cols_poly[0])
        ax.plot(
            cols_poly,
            rows_poly,
            color="red",
            linewidth=2,
            label="true containing polygon",
        )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("col")
    ax.set_ylabel("row")
    if title:
        ax.set_title(title)

    if save_path:  # for saving the image
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def apply_model_and_plot(
    model,
    center_coords,
    ndvi,
    s_hist,
    i_hist,
    threshold,
    dist_per_cell_meters=30,
    device="cpu",
    fp=0.01,
    fn=0.01,
    plantspergrid=10,
    cmap="hot",
    title=None,
    save_path=None,
    ax=None,
):
    """
    function to apply the trained model to current simulation data
    and visualize the predicted infected area

    inputs
    --------
    model = trained OutbreakSTGNN model
    center_coords = tuple (lat, lon) for the center of the grid
    ndvi = (n_rows, n_cols) NDVI array used for model predictions
    s_hist = (T, n_rows, n_cols) historical susceptible matrices
    i_hist = (T, n_rows, n_cols) historical infected matrices
    threshold = infection probability cutoff for predicted infected area
    dist_per_cell_meters = real-world size of one pixel in meters
    device = device to run model inference on ("cpu" or "cuda")
    fp = false positive rate for sighting reports
    fn = false negative rate for sighting reports
    plantspergrid = number of plants per grid cell for report generation
    cmap = colormap for the probability heatmap
    title = custom title for the plot
    save_path = location to save the figure
    ax = existing matplotlib axis to plot on

    returns
    -------
    fig = matplotlib figure object
    predictions = (T, n_rows, n_cols) predicted infection probabilities
    georef = GridGeoref object for coordinate conversions
    polygon = predicted containment polygon as list of (lat, lon) tuples
    """
    model.eval()

    n_rows, n_cols = ndvi.shape
    timesteps = len(i_hist)

    # create georef and location matrix
    georef = GridGeoref(
        center_coords=center_coords,
        dist_per_cell=dist_per_cell_meters,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    locs = generate_latlon_matrix(
        ndvi, center_coords, dist_per_cell=dist_per_cell_meters
    )

    # generate sightings from historical simulation data
    dist_per_grid_degrees = georef.lat_scale
    sightings = []
    timestamps = []

    for day in range(timesteps):
        reports = genReports(
            s_hist[day],
            i_hist[day],
            day=day,
            locs=locs,
            dist_per_grid=dist_per_grid_degrees,
            seed=None,
            fp=fp,
            fn=fn,
            plantspergrid=plantspergrid,
        )
        sightings += reports_to_sightings(reports)
        timestamps.append(float(day))

    # build episode tensors for model input
    x_seq, _ = build_episode_tensors(
        ndvi, sightings, np.array(timestamps), georef, None
    )
    x_seq = x_seq.to(device)

    # build graph connections
    edge_index, edge_weight = build_connections(n_rows, n_cols)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # run model to get predictions
    with torch.no_grad():
        logits_list = model(x_seq, edge_index, edge_weight)

    # convert logits to probabilities
    predictions = torch.sigmoid(torch.stack(logits_list)).cpu().numpy()
    predictions = predictions.reshape(timesteps, n_rows, n_cols)

    # use the last timestep (current state) for plotting
    current_probs = predictions[-1]

    # create polygon around predicted infected area
    polygon = create_polygon(
        current_probs,
        georef,
        threshold=threshold,
        method="contour",
        simplify_tolerance_cells=0.5,
    )

    # create the plot
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # plot probability heatmap
    im = ax.imshow(current_probs, cmap=cmap, vmin=0, vmax=1, origin="upper")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Infection probability")

    # overlay polygon if it exists
    if polygon is not None and len(polygon) > 0:
        # convert lat/lon polygon to row/col indices
        rows_poly, cols_poly = [], []
        for lat, lon in polygon:
            row = georef.center_row - (lat - georef.center_lat) / georef.lat_scale
            col = georef.center_col + (lon - georef.center_lon) / georef.lon_scale
            rows_poly.append(row)
            cols_poly.append(col)

        # close the loop back to the first vertex
        rows_poly.append(rows_poly[0])
        cols_poly.append(cols_poly[0])

        ax.plot(
            cols_poly,
            rows_poly,
            color="cyan",
            linewidth=2,
            linestyle="--",
            label="Predicted infected area",
        )
        ax.legend(loc="upper right", fontsize=8)

    # add labels and title
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Predicted infection spread (threshold={threshold:.2f})")

    # save if requested
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, predictions, georef, polygon


N_FEATURES = 4  # ndvi, all_sighting, time_since_sighting, dist_to_sighting


def sightings_to_grid_timeline(sightings, georef, timestamps):
    """
    Optimized version using KD-tree for fast nearest-neighbor distance computation.
    """
    n_nodes = georef.n_rows * georef.n_cols
    T = len(timestamps)
    first_sighting_time = np.full(n_nodes, np.inf)

    for s in sightings:
        row, col = georef.latlon_to_rc(s["lat"], s["lon"])
        node = georef.node_id(row, col)
        first_sighting_time[node] = min(first_sighting_time[node], s["t"])

    all_sighting = np.zeros((T, n_nodes), dtype=np.float32)
    time_since = np.zeros((T, n_nodes), dtype=np.float32)
    dist_to_sighting = np.full((T, n_nodes), 1e3, dtype=np.float32)

    # Pre-compute all node coordinates once
    rows = np.arange(georef.n_rows)
    cols = np.arange(georef.n_cols)
    grid_r, grid_c = np.meshgrid(rows, cols, indexing="ij")
    all_node_coords = np.column_stack([grid_r.ravel(), grid_c.ravel()])

    for t_idx, t in enumerate(timestamps):
        active_mask = first_sighting_time <= t
        all_sighting[t_idx] = active_mask.astype(np.float32)
        time_since[t_idx] = np.where(
            active_mask, np.maximum(t - first_sighting_time, 0.0), 0.0
        )

        if active_mask.any():
            active_nodes = np.nonzero(active_mask)[0]
            active_coords = all_node_coords[
                active_nodes
            ]  # Direct indexing, no conversion needed

            # KD-tree for fast nearest neighbor search (Chebyshev distance)
            tree = cKDTree(active_coords)
            dist, _ = tree.query(all_node_coords, k=1, p=np.inf)  # p=np.inf = Chebyshev
            dist_to_sighting[t_idx] = dist

    return all_sighting, time_since, dist_to_sighting


def build_episode_tensors(
    ndvi_grid, sightings, timestamps, georef, ground_truth_masks=None
):
    """
    Assembles one episode into the tensors OutbreakSTGNN.forward() expects.

    ndvi_grid: (n_rows, n_cols) static NDVI, values in [0, 1]
    sightings: cumulative sighting list, see sightings_to_grid_timeline()
    timestamps: sorted array of T snapshot times
    georef: the GridGeoref for this episode's grid
    ground_truth_masks: optional (T, n_rows, n_cols) binary array of true
        infection state per day, from your simulator. Required for
        training; omit for inference.

    Returns:
        x_seq: (T, n_nodes, N_FEATURES) float32 tensor
        y_seq: (T, n_nodes) float32 tensor, or None if ground_truth_masks
               wasn't given
    """
    n_rows, n_cols = georef.n_rows, georef.n_cols
    n_nodes = n_rows * n_cols
    T = len(timestamps)

    all_sighting, time_since, dist_to_sighting = sightings_to_grid_timeline(
        sightings, georef, timestamps
    )

    ndvi_flat = ndvi_grid.reshape(-1).astype(np.float32)
    time_since_norm = time_since / (time_since.max() + 1e-6)
    dist_norm = np.clip(dist_to_sighting / max(n_rows, n_cols), 0, 1)

    features = np.zeros((T, n_nodes, N_FEATURES), dtype=np.float32)
    for t in range(T):
        features[t, :, 0] = ndvi_flat
        features[t, :, 1] = all_sighting[t]
        features[t, :, 2] = time_since_norm[t]
        features[t, :, 3] = dist_norm[t]

    x_seq = torch.tensor(features, dtype=torch.float32)

    y_seq = None
    if ground_truth_masks is not None:
        y_seq = torch.tensor(
            ground_truth_masks.reshape(T, n_nodes).astype(np.float32),
            dtype=torch.float32,
        )

    return x_seq, y_seq
