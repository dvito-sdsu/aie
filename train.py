import math
import numpy as np
import random
from scipy.signal import convolve2d
from skimage import measure
from shapely.geometry import Polygon, MultiPoint


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
    ] = 1

    # 2. Create 2 additional arrays of zeros with the same shape
    S = np.zeros_like(input_np) + 1 - I
    R = np.zeros_like(input_np)

    # 3. Return all three newly created arrays
    return S, I, R


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
    infected_pressure = convolve2d(I, conv_matrix, mode="same", boundary="fill")

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


def generate_wind_kernel(radius=2, sigma=1.0, wind_x=0.0, wind_y=0.0):
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
    size = 2 * radius + 1

    # Create a coordinate grid centered at (0,0)
    # creating the array
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)

    # calculating the probabilities with the wind
    kernel = np.exp(-((xx - wind_x) ** 2 + (yy - wind_y) ** 2) / (2 * sigma**2))

    # return and normalize
    return kernel / np.sum(kernel)


def randomBG(choice="none"):
    """
    Function to randomly choose beta and gamma

    inputs

    choice (optional) - override the random selection of disease spreadability
        TODO(?) - maybe choose specific values of beta or gamma?
        might defeat the purpose of this function though

    returns

    beta - variable for the spread rate of the disease
    gamma - rate the infected individuals are removed from the infected category

    """
    # Tiered sampling to get an even mix of slow, medium, and fast diseases
    if choice in ["slow", "medium", "fast"]:
        disease_type = choice
    else:
        disease_type = random.choice(["slow", "medium", "fast"])
    while True:
        if disease_type == "slow":
            beta = random.uniform(0.01, 0.08)
            gamma = random.uniform(0.005, 0.04)
        elif disease_type == "medium":
            beta = random.uniform(0.08, 0.20)
            gamma = random.uniform(0.04, 0.10)
        else:  # fast
            beta = random.uniform(0.20, 0.60)
            gamma = random.uniform(0.10, 0.30)

        # ensure randomly chosen beta and gamma are viable outbreak values
        if beta / gamma > 1.0:
            return beta, gamma


def randomFloat(min=0, max=4):
    """
    random float within range min to max with equal chances between the range
    """
    return random.random() * random.randint(1, max - min) + min


def random_wind_kernel(choice="None"):
    """
    function to create a random wind kernel for the simulation
    will use generate_wind_kernel to create it, this function is just for the randomization

    random value ranges are arbitrary

    inputs
    -------

    choice = #TODO might be removed


    returns
    -----
    conv matrix = randomly generated convulational matrix
    """
    radius = random.randint(3, 7)
    sigma = randomFloat(1, 4)
    wind_x = randomFloat(0, 4)
    wind_y = randomFloat(0, 4)

    return generate_wind_kernel(
        radius=radius, sigma=sigma, wind_x=wind_x, wind_y=wind_y
    )


def randomSimulation(ndvi, timesteps=50):
    """
    function to create and store a random simulation

    inputs
    --------
    ndvi = ndvi to base the simulation on
            will likely come from random ndvi function
    timesteps = total number of timesteps to calculate

    returns #TODO
    -------
    s_hist = history of susceptible matrix has shape ndvi X timesteps
    i_hist = history of infected matrix has shape ndvi X timesteps
    """

    if timesteps < 10:
        raise ValueError(
            f"please choose more than 10 timesteps, input {timesteps} timesteps"
        )
    elif timesteps > 500:
        raise ValueError(
            f"please choose less than 500 timesteps, input {timesteps} timesteps"
        )

    beta, gamma = randomBG()
    conv_matrix = random_wind_kernel()
    S, I, R = start_SIR_random_middle(ndvi)

    S, I, R = sir_spatial_step(S, I, R, ndvi, conv_matrix, beta, gamma)

    s_hist = S.copy()
    i_hist = I.copy()

    for t in range(timesteps - 1):
        S, I, R = sir_spatial_step(S, I, R, ndvi, conv_matrix, beta, gamma)
        s_hist = np.vstack(s_hist, S)
        i_hist = np.vstack(i_hist, I)
    return s_hist, i_hist


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
