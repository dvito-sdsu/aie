# %%
import numpy as np
import rasterio
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import time
from skimage import measure
from shapely.geometry import Polygon, MultiPoint
import math

# %%
# importing library to convert notebook to web page
import streamlit as st

# %%
#importing other notebook
import importnb
with importnb.Notebook():
    from ndvi import get_landsat_ndvi_matrix



# %% [markdown]
# The below function `conv_kernal` is meant to create a different covolutional matrix for the sir model, one infected square can, by default, spread up to 3 grids away, with % decreasing in accordance to inverse square law. <br>
# Currently this kernel is using aribtrary values. Before real life application, ideally each disease this cnn will be trained to detect has a dedicated CK for it and diseases with similar spreading capabilities. Future improvements could include wind or water changing the spread likelihood.

# %%
def conv_kernal(radius = 3):
  """
  creates a ck with size 2*radius + 1
  this allows infections to spread up to 3 tiles away
  """
  size = 2 * radius + 1
  kernel = np.zeros((size,size))
  for y in range(size):
    for x in range(size):
      if x == radius and y == radius:
        kernel[y,x] = 0 # plants cannot reinfect themselves
        continue

# %%
def generate_wind_kernel(radius = 2, sigma=1.0, wind_x=0.0, wind_y=0.0):
    """
    convolutional matrix but with a wind vector

    inputs
    radius - distance a disease can spread
    sigma - spread of the disease through the wind
    wind_x - horizontal component of wind (0 blows east (right))
    wind_y - vertical component of wind (0 blows north (up))

    returns
    conv_matrix
    """

    # radius of 2 = 5x5 conv matrix
    size = 2*radius + 1


    # Create a coordinate grid centered at (0,0)
    # creating the array
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)

    # calculating the probabilities with the wind
    kernel = np.exp(-((xx - wind_x)**2 + (yy - wind_y)**2) / (2 * sigma**2))

    # return and normalize
    return kernel / np.sum(kernel)


# %% [markdown]
# Below is the final simulation code, what is changed above should not alter the core structure of the simulation step function.

# %%
def sir_spatial_step(S, I, R, ndvi, conv_matrix, beta, gamma, dt=1.0):
    """
    single step in the simulation

    inputs

    S, I, R - numpy arrays of same shape
    ndvi - array of plant begetation density(influences spreadability)
    conv_matrix convulutional matrix for optimized disease spread
    beta - transmission rate
    gamma - Recovery/removal rate (not able to be reinfected)
    dt - time step size (default 1)

    returns
    S_next, I_next, R_next - next step of the simulation
    """
    # calculating base chance for tiles to be infected
    # same - do not change the output shape
    # fill - assumes nearby vegetation is not undergoing the same disease yet
    infected_pressure = convolve2d(I, conv_matrix, mode='same', boundary='fill')


    # adjusting transmission with beta, ndvi, and time using susceptibile array
    new_infections = beta * S * ndvi * infected_pressure * dt

    # calculating recoveries with gamma and dt
    new_recoveries = gamma * I * dt * 0

    # making sure not to infect or recover more than what is available in a cell
    # ensures further calculations do not go negative
    new_infections = np.minimum(new_infections, S)
    new_recoveries = np.minimum(new_recoveries, I)

    # saving the new SIR numpy arrays
    S_next = S - new_infections
    I_next = I + new_infections - new_recoveries
    R_next = R + new_recoveries

    # realign values to our ranges
    S_next = np.clip(S_next, 0.0, 1.0)
    I_next = np.clip(I_next, 0.0, 1.0)
    R_next = np.clip(R_next, 0.0, 1.0)

    return S_next, I_next, R_next

# %%
def start_SIR_random(input_array):
    """
    1) Selects a random element in the input array and changes its value to 1.
    2) Creates two additional arrays of zeros with the exact same shape.
    3) Returns all three arrays.

    Parameters:
    input_array (list or np.ndarray): The source array to modify.

    Returns:
    tuple: (modified_original_array, zero_array_1, zero_array_2)
    """
    # Convert input to a numpy array just to easily extract its shape safely
    input_np = np.asarray(input_array)

    # 1. Create the first new array of zeros and set a random element to 1
    I = np.zeros_like(input_np)
    rando = I.shape

    # Pick a random flat index and map it back to the multi-dimensional shape
    I[np.random.randint(0,rando[0]-1)][np.random.randint(0,rando[1] -1)] = 1

    # 2. Create 2 additional arrays of zeros with the same shape
    S = np.zeros_like(input_np) + 1 - I
    R = np.zeros_like(input_np)

    # 3. Return all three newly created arrays
    return S, I, R

def start_SIR_random_middle(input_array):
    """
    Creates 3 arrays,
    one with the initial infection,
    one of ones except the initial infected tile,
    and an empty one,
    all of shape input_array

    chooses the middle 1/9 of the input array

    Parameters:
    input_array (list or np.ndarray): The source array to modify.

    Returns:
    S,I,R
    """
    # Convert input to a numpy array just to easily extract its shape safely
    input_np = np.asarray(input_array)

    # 1. Create the first new array of zeros and set a random element to 1
    I = np.zeros_like(input_np)
    rando = I.shape

    # Pick a random flat index and map it back to the multi-dimensional shape
    I[
        np.random.randint(
            (math.floor(rando[0] / 3)), 2 * (math.floor(rando[0] / 3)) - 1
        )
    ][
        np.random.randint(
            (math.floor(rando[0] / 3)), 2 * (math.floor(rando[0] / 3)) - 1
        )
    ] = .2

    # 2. Create 2 additional arrays of zeros with the same shape
    S = np.zeros_like(input_np) + 1 - I
    R = np.zeros_like(input_np)

    # 3. Return all three newly created arrays
    return S, I, R

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
            cols_poly, rows_poly, color="red", linewidth=2, label="True polygon"
        )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("col")
    ax.set_ylabel("row")
    if title:
        ax.set_title(title)

    if save_path:  # for saving the image
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig




