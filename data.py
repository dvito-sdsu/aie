# TODO imports
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import os
import random


def genReports(
    Sus,
    Infected,
    day,
    locs,
    dist_per_grid,
    seed=None,
    fp=0.01,
    fn=0.01,
    plantspergrid=10,
):
    """
    goal
    change simulation data into realistic reports of format [long,lat]
    maybe date as well?


    INPUTS
    --------
    sus = susceptible matrix of size MxN floats in range [0,1]
    infected = infected matrix of size MxN floats in range [0,1]

    day = how many time steps have occurred

    locs = 3d array from generate_latlon_matrix
    dist_per_grid = add random jitter to locations

    seed = rng seed for reproducibility

    fp = false positve rate for reports, should be randomized during the training after each indivdual simulation (TBD range)
    fn = false negative, same as above

    plantspergrid = how many plants are contained within each grid of the ndvi matrix

    RETURNS
    ---------
    ?x3 matrix of reported disease detections with format LAT LON DAY
    """
    rng = np.random.default_rng(seed)

    # true positives

    counts = rng.poisson(Infected * plantspergrid)
    rows, cols = np.nonzero(counts)
    n = counts[rows, cols]
    rows_rep = np.repeat(rows, n)
    cols_rep = np.repeat(cols, n)

    # removing false negatives
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

    # combining positives
    all_rows = np.concatenate([rows_rep, fp_rows_rep])
    all_cols = np.concatenate([cols_rep, fp_cols_rep])

    if len(all_rows) == 0:
        return np.empty((0, 3), dtype=float)

    # shuffling
    perm = rng.permutation(len(all_rows))
    all_rows = all_rows[perm]
    all_cols = all_cols[perm]

    # locs is (M, N, 2) with the last axis ordered [lat, lon]
    lat = locs[all_rows, all_cols, 0].astype(float)
    lon = locs[all_rows, all_cols, 1].astype(float)

    # adding jitter in each grid
    lat = lat + rng.uniform(-dist_per_grid / 2, dist_per_grid / 2, size=lat.shape)
    lon = lon + rng.uniform(-dist_per_grid / 2, dist_per_grid / 2, size=lon.shape)

    day_col = np.full(lat.shape, day, dtype=float)

    reports = np.column_stack([lat, lon, day_col])
    return reports


def reports_to_sightings(reports):
    # small function to convert the output from genReports() to a format readable by the st-gnn
    # can be implemented into genReports later
    return [{"lat": r[0], "lon": r[1], "t": r[2]} for r in reports]


def generate_latlon_matrix(input_matrix, center_idx, center_coords, dist_per_cell):
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
    center_row, center_col = center_idx
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

    if used is None:
        used = []

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
    ndvi = np.array(img).astype(float)

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
    lat_str, lon_str = name.split("-")
    return float(lat_str), float(lon_str)
