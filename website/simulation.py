# %%
import numpy as np
import rasterio
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import time

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
    new_recoveries = gamma * I * dt

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




