import numpy as np
import rasterio
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import streamlit as st
from ndvi import get_landsat_ndvi_matrix
from simulation import sir_spatial_step, generate_wind_kernel, start_SIR_random
import time

st.set_page_config(layout="wide")


# functions
def get_NDVI(lati, longi, dist):

    status_placeholder = st.empty()

    status_placeholder.info("Attempting to get data from API...")

    time.sleep(1)
    try:
        ndvi = get_landsat_ndvi_matrix(lati, longi, half_side_meters=dist)
        st.session_state["ndvi"] = ndvi
        st.write("Got Ndvi")
        st.session_state["lati"] = lati
        st.session_state["long"] = long
        st.session_state["dist"] = dist
        del st.session_state["conv_matrix"]

        status_placeholder.success("Data Acquired")
    except Exception as e:
        print(f"An error occurred: {e}")
        status_placeholder.error(f"Failed to fetch data. Error {e}")
    pass


def init_matrix(spreadability, variance, wind_x, wind_y, magni_mult):

    status_placeholder = st.empty()
    ndvi = st.session_state["ndvi"]
    status_placeholder.info("Calculating Kernel")
    conv_matrix = generate_wind_kernel(
        radius=spreadability,
        sigma=variance,
        wind_x=wind_x * magni_mult,
        wind_y=-wind_y * magni_mult,
    )
    S, I, R = start_SIR_random(ndvi)
    st.session_state["var"] = variance
    st.session_state["spread"] = spreadability
    st.session_state["S"] = S
    st.session_state["I"] = I
    st.session_state["R"] = R
    st.session_state["conv_matrix"] = conv_matrix
    st.session_state["timetotal"] = 0
    try:  # to differentiate between function being run with or without the code block being displayed
        st.session_state["beta"] = beta
        st.session_state["gamma"] = gamma
        st.session_state["wind_x"] = wind_x
        st.session_state["wind_y"] = wind_y
        st.session_state["magni_mult"] = magni_mult
    except Exception as e:
        pass
    status_placeholder.success("Finished")
    pass


def simulate_steps(timestep):

    status_placeholder = st.empty()
    ndvi = st.session_state["ndvi"]
    status_placeholder.info("Starting")
    timetotal = st.session_state["timetotal"]
    conv_matrix = st.session_state["conv_matrix"]
    S = st.session_state["S"]
    I = st.session_state["I"]
    R = st.session_state["R"]
    for t in range(timestep):
        timetotal += 1
        S, I, R = sir_spatial_step(
            S,
            I,
            R,
            ndvi,
            conv_matrix,
            st.session_state["beta"],
            st.session_state["gamma"],
        )
    status_placeholder.success("Finished")
    st.session_state["S"] = S
    st.session_state["I"] = I
    st.session_state["R"] = R

    rgb_grid = np.stack([I, R, S], axis=-1)

    fig2, ax2 = plt.subplots(figsize=(8, 8))

    ax2.imshow(rgb_grid, origin="upper")

    ax2.set_title("Current simulation", fontsize=14, pad=15)
    ax2.axis("off")

    legend_elements = [
        Patch(facecolor="blue", label="Susceptible (S)"),
        Patch(facecolor="red", label="Infectious (I)"),
        Patch(facecolor="green", label="Recovered (R)"),
    ]

    ax2.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    # Apply tight layout formatting to the overall figure
    fig2.tight_layout()
    st.session_state["Simulation_graph"] = fig2
    st.session_state["Simulation_ax"] = ax2

    # st.pyplot(fig2)


# Hides the declaration block of the website
def toggle_layout():
    st.session_state.show_all = not st.session_state.show_all


st.title("NDVI based crop SIR Simulation")
# code block below is to make sure the variables are saved when the declaration code is toggled
if "ndvi" not in st.session_state:
    st.session_state["ndvi"] = np.load("Default_ndvi_data.npy")
if "wind_x" not in st.session_state:
    st.session_state["wind_x"] = 0.0
if "wind_y" not in st.session_state:
    st.session_state["wind_y"] = 0.0
if "magni_mult" not in st.session_state:
    st.session_state["magni_mult"] = 1.0
if "beta" not in st.session_state:
    st.session_state["beta"] = 0.3
if "gamma" not in st.session_state:
    st.session_state["gamma"] = 0.005
if "show_all" not in st.session_state:
    st.session_state.show_all = True
if "lati" not in st.session_state:
    st.session_state["lati"] = 37.498056
if "long" not in st.session_state:
    st.session_state["long"] = -120.812583
if "dist" not in st.session_state:
    st.session_state["dist"] = 1500
if "spread" not in st.session_state:
    st.session_state["spread"] = 5
if "var" not in st.session_state:
    st.session_state["var"] = 3


if st.session_state.show_all:

    st.header("Declare Variables")

    # satellite data init
    sat_init_1, sat_init_2, sat_init_3 = st.columns(3)

    with sat_init_1:
        lat = st.number_input("Latitude", value=st.session_state["lati"])
    with sat_init_2:
        long = st.number_input("Longitude", value=st.session_state["long"])
    with sat_init_3:
        dist = st.number_input("Range", value=st.session_state["dist"])

    # for creating the conv matrix
    mat1, mat2, mat3, mat4 = st.columns(4)
    with mat1:
        spreadability = st.number_input(
            "maximum spread range", value=st.session_state["spread"]
        )
    with mat2:
        variance = st.number_input(
            "spread concentration", value=st.session_state["var"]
        )
    with mat3:
        beta = st.number_input(
            "Rate of transmission",
            value=st.session_state["beta"],
            min_value=0.0,
            max_value=1.0,
        )
    with mat4:
        gamma = st.number_input(
            "Rate of removal",
            value=st.session_state["gamma"],
            min_value=0.0,
            max_value=1.0,
        )

    # slider input test
    wind1, wind2, wind3, wind4 = st.columns(4)
    with wind1:
        wind_x = st.slider(
            "Wind Horizontal Component",
            min_value=-1.0,
            max_value=1.0,
            value=st.session_state["wind_x"],
            step=0.01,
        )
    with wind2:
        wind_y = st.slider(
            "Wind Horizontal component",
            min_value=-1.0,
            max_value=1.0,
            value=st.session_state["wind_y"],
            step=0.01,
        )
    with wind3:
        magni_mult = st.slider(
            "Magnitude (useless without wind vector)",
            min_value=0.1,
            max_value=5.0,
            value=st.session_state["magni_mult"],
            step=0.1,
        )

    # normalizing
    magnitude = np.sqrt(wind_x**2 + wind_y**2)

    if magnitude > 0:
        st.session_state["norm_x"] = wind_x / magnitude
        st.session_state["norm_y"] = wind_y / magnitude
    else:
        st.session_state["norm_x"], st.session_state["norm_y"] = 0.0, 0.0

    # creating unit fig diagram
    unitfig, unitax = plt.subplots(figsize=(5, 5))

    # drawing unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    unitax.plot(
        np.cos(theta),
        np.sin(theta),
        color="lightgray",
        linestyle="--",
        label="Unit Circle",
    )

    # highlighting axes
    unitax.axhline(0, color="black", linewidth=0.8)
    unitax.axvline(0, color="black", linewidth=0.8)

    # drawing the vector

    unitax.quiver(
        0,
        0,
        st.session_state["norm_x"],
        st.session_state["norm_y"],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="royalblue",
        width=0.015,
    )

    # small dot at the tip
    unitax.plot(
        st.session_state["norm_x"],
        st.session_state["norm_y"],
        marker="o",
        color="royalblue",
        markersize=6,
    )

    # formatting the plot
    unitax.set_xlim([-1.2, 1.2])
    unitax.set_ylim([-1.2, 1.2])
    unitax.set_aspect(
        "equal"
    )  # Forces the plot to be a perfect square, preventing oval distortion
    unitax.grid(True, which="both", linestyle=":", alpha=0.5)
    unitax.set_title("Resultant Unit Vector", fontsize=12)

    #  displaying info sideby side
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.metric("Normalized X", f"{st.session_state["norm_x"]:.5f}")
        st.metric("Normalized Y", f"{st.session_state["norm_y"]:.5f}")

        show_graph = st.toggle("Toggle Vector Graph", value=True)

    with col2:
        # Use st.pyplot to render the Matplotlib figure directly
        if show_graph:
            st.pyplot(unitfig)


    simcol1, simcol2, simcol3 = st.columns([0.25, 0.25, 0.5])

    with simcol1:
        if st.button(
            f"Get NDVI data at {st.session_state["lati"]}, {st.session_state["long"]}"
        ):
            get_NDVI(lat, long, dist)
    with simcol2:
        if st.button("Initialize Spreading Capabilities"):
            init_matrix(
                spreadability,
                variance,
                st.session_state["norm_x"],
                st.session_state["norm_y"],
                magni_mult,
            )
    # with simcol3:
    #     if st.button(f"Simulate {timestep} timesteps"):
    #         simulate_steps(timestep)

tab1, tab2 = st.tabs(["Satellite NDVI View", "SIR Simulation View"])
fig, ax = plt.subplots()

with tab1:
    st.subheader("NDVI Display")
    st.button("Show/Hide Variable Declaration", on_click=toggle_layout)

    graph_col, padding_col = st.columns([0.5, 0.5])

    with graph_col:

        ax.imshow(st.session_state["ndvi"], vmin=0.0, origin="upper")
        st.session_state["ndvi_fig"] = fig
        st.session_state["ndvi_ax"] = ax
        st.pyplot(fig)

with tab2:
    st.subheader("SIR Simulation")
    select_sim_col, sim_col, filer_col = st.columns([0.25, 0.25, 0.5])

    with select_sim_col:
        timestep = st.number_input("Number of time steps to calculate", value=10)
    with sim_col:
        if st.button(f"Simulate {timestep} Timesteps"):
            if "conv_matrix" not in st.session_state:
                init_matrix(
                    st.session_state["spread"],
                    st.session_state["var"],
                    st.session_state["norm_x"],
                    st.session_state["norm_y"],
                    st.session_state["magni_mult"],
                )
            simulate_steps(timestep)

    graph_col, padding_col = st.columns([0.5, 0.5])

    with graph_col:
        if "Simulation_graph" in st.session_state:
            fig2 = st.session_state["Simulation_graph"]
            ax2 = st.session_state["Simulation_ax"]
            st.pyplot(fig2)
