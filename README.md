# AIE: Plant Disease Outbreak Simulation and Prediction

AIE simulates how a plant disease spreads across farmland and trains neural networks to predict where an outbreak is, given noisy sighting reports. It uses satellite vegetation data (NDVI) as the landscape, a spatial SIR-style model to generate synthetic outbreaks, and spatiotemporal models (a graph network and U-Net variants) to learn from them.

## How it works

```
Landsat imagery ──► NDVI grid ──► spatial SIR simulation ──► daily sighting reports
(Planetary Computer)   (ndvi.py)      (sim_class.py)          (with false +/-)
                                                                     │
                                  prediction maps ◄── model ◄── training episodes
                                                    (model.py)    (train.py, data.py)
```

1. **Landscape.** NDVI (a vegetation density index) is computed from Landsat red and near-infrared bands for a square area around a coordinate.
2. **Simulation.** A grid-based SIR model (Susceptible, Infected, Recovered) spreads infection between cells using a Gaussian dispersal kernel, which can be shifted to mimic wind. Denser vegetation (higher NDVI) increases infection pressure.
3. **Reports.** Each simulated day produces geolocated sighting reports (`[lat, lon, day]`), drawn as Poisson counts from the infected density, with configurable false-positive and false-negative rates. This mimics a detection system such as drone imagery plus image recognition.
4. **Learning.** The models take the sighting stream and predict, per grid cell and per day, whether the cell is truly infected. Ground truth is the simulated infection state thresholded at a chosen fraction.

## Repository layout

| Path | Purpose |
|---|---|
| `ndvi.py` | Fetches Landsat Collection 2 Level-2 data from Microsoft Planetary Computer and computes NDVI (and optionally surface temperature). |
| `sim_class.py` | `simModel`: the spatial SIR simulator, report generation, and save/load (`.npz`, `.h5`). |
| `model.py` | Model definitions: `OutbreakSTGNN`, `OutbreakUNet`, `ConvLSTMUNet`. (only `UNet` is used) |
| `train.py` | Episode generation, loss functions, training loop, checkpoint save/load. |
| `data.py` | Report generation, feature building, graph construction, and NDVI tile loading used by `train.py`. |
| `georef.py` | `GridGeoref`: converts between grid cells and latitude/longitude. |
| `unet.py` | U-Net related code. |
| `simulation.py`, `simulation.ipynb` | Simulation scripts and notebook. |
| `collectndvi.ipynb`, `ndvi.ipynb` | Notebooks for collecting and exploring NDVI data. |
| `webtest.py`, `location_to_website.html`, `website/` | Web front end (the repo lists `streamlit` as a dependency). |
| `farmlocations/` | Farm location data. |
| `models/` | Model files. |
| `aietables.sql`, `aieconstraints.sql`, `aiesampledata.sql` | Database schema, constraints, and sample data. |
| `Default_ndvi_data.npy` | Default NDVI array. |
| `requirements.txt` | Python dependencies. |

## Installation

Requires Python 3.9 or newer (Python 3.10+ recommended).

```bash
git clone https://github.com/dvito-sdsu/aie.git
cd aie
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

`requirements.txt` contains a line that pip cannot parse (`torch --index-url ...`) plus several duplicate entries, so install PyTorch separately first, then the rest:

```bash
# CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Everything else
pip install numpy scipy matplotlib pandas h5py scikit-image shapely rasterio \
            rioxarray planetary_computer pystac_client streamlit importnb
```

For GPU training, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) instead.

## Usage

### 1. Get an NDVI grid

```python
from ndvi import get_landsat_ndvi_matrix

# half_side_meters is the distance from the center to the edge of the square.
# Landsat pixels are ~30 m, so expect roughly (2 * half_side_meters / 30) cells per side.
ndvi = get_landsat_ndvi_matrix(37.498056, -120.812583, half_side_meters=1500)
print(ndvi.shape)
```

This searches the five most recent Landsat scenes with under 10% cloud cover, picks the newest, and returns NDVI clipped to `[0, 1]`. It needs internet access and raises `ValueError` if no scene matches.

`get_landsat_ndvi_and_temp_matrix(...)` does the same and also returns surface temperature in °C.

### 2. Run a simulation

```python
from sim_class import simModel

lat, lon = 37.498056, -120.812583

sim = simModel(
    ndvi,
    center_coords=(lat, lon),
    dist_per_grid=30,     # grid cell size in meters
    fp=0.01,              # false-positive rate for reports
    fn=0.01,              # false-negative rate for reports
    total_steps=50,       # preallocate history (fast; good for bulk data generation)
)

sim.randomBG("fast")      # sample beta/gamma for an archetype (see below)
sim.randomMatrix()        # random Gaussian dispersal kernel (or use createMatrix)
sim.initSim()             # one infected cell near the middle of the grid
sim.complete_sim()        # run all remaining steps

sim.S_hist, sim.I_hist    # (T, rows, cols) susceptible / infected history
sim.reports               # (N, 3) array of [lat, lon, day]
```

Without `total_steps`, the simulator stores history as it goes and you advance it with `sim.step(n)`.

**Disease archetypes** for `randomBG(choice)`:

| Archetype | Beta (spread) | Gamma (removal) | Behavior |
|---|---|---|---|
| `unviable` | 0.30 to 0.60 | 0.30 to 0.60 | Dies out before causing an outbreak |
| `slow` | 0.05 to 0.15 | 0.01 to 0.04 | Persistent but slow |
| `normal` | 0.15 to 0.35 | 0.02 to 0.08 | Sustained spread |
| `fast` | 0.35 to 0.70 | 0.08 to 0.20 | Rapid, visible outbreak |

If `choice` is not one of these, one of `slow`, `normal`, or `fast` is picked at random. Unless you create the simulator with `recovery=True`, `randomBG` sets gamma to 0, making the model effectively SI (infected cells never recover). You can also set rates directly with `sim.setBG(beta, gamma)`, or pass both `beta` and `gamma` to the constructor.

**Saving and loading:**

```python
path = sim.save("simdata")                    # compressed .npz (reports, NDVI, ground truth, metadata)
sim2 = simModel.load(path)

path = sim.save_h5("simdata")                 # HDF5 with full S/I history, for training
sim3 = simModel.load_h5(path)
```

`save` writes ground truth as `I_hist > infected_threshold` (default 0.1). `save_h5` requires at least one step to have run. HDF5 files are named with a random UUID.

### 3. Train a model

```python
from train import train, save_model

model = train(
    ndvi_folder="brazilFarms",     # folder of NDVI tiles that data.py loads from
    n_epochs=20,
    episodes_per_epoch=10,
    timesteps=50,
    hidden_channels=256,
    dist_per_cell_meters=30,
    device="cpu",                  # or "cuda"
)
```

Each episode loads a random NDVI tile, runs a fresh random outbreak on it, generates daily reports, converts everything to tensors, and takes one optimizer step. Timesteps are processed in chunks (`chunk_size=10`) with gradients accumulated across chunks to limit memory use. The default loss is Tversky loss (`alpha=0.8`, `beta=0.2`), which penalizes missed infections more than false alarms.

Checkpoints are saved and restored with `save_model(...)` and `load_model(filepath, device)`.

## Models (`model.py`)

| Model | Input | Description |
|---|---|---|
| `OutbreakSTGNN` | `(T, N, F)` node features plus edge index and weights | Graph model. Each timestep is processed independently with shared weights and message-passing aggregation. No recurrence, so it needs far less memory than a DCRNN. Currently the model used by `train.py`. |
| `OutbreakUNet` | `(T, C, H, W)` | U-Net on raster inputs, treating timesteps as a batch. Temporal information comes from the input features. |
| `ConvLSTMUNet` | `(B, T, C, H, W)` | U-Net with a ConvLSTM at the bottleneck to carry state across timesteps. |

All models return a list of `T` tensors of raw logits (apply a sigmoid for probabilities).

## Simulation details

- **Infection update.** Each day, infection pressure is the infected grid convolved with the dispersal kernel. New infections are `beta * S * NDVI * pressure`, capped at the available susceptible amount. Removals are `gamma * I`.
- **Initial infection.** One cell in the middle third of the grid is set to 0.2 infected, preferring cells with NDVI above 0.3.
- **Reports.** True positives are Poisson draws from `I * NDVI`, thinned by the false-negative rate. False positives are Poisson draws from `S * NDVI * fp`. Report coordinates are jittered within their cell.
- **Georeferencing.** Grid cells are mapped to latitude/longitude using an equirectangular approximation around the center coordinate.
- **NDVI scaling.** `train.py` rescales NDVI with `(0.5 + ndvi * 1.5) ** 2` before simulating. `simModel` uses the NDVI exactly as you pass it, so apply any scaling yourself if you want the same behavior.

## Known limitations

- `simModel.initSim(mode="random")` calls `randomStart()`, which is not defined. Use the default `mode="middle"`.
- `load_h5` does not restore the R history or the dispersal kernel. Call `createMatrix()` or `randomMatrix()` before stepping a loaded simulation further.
- `train.py` has its own copies of the simulation functions, and its `randomBG` always forces the `fast` archetype. `sim_class.py` does not do this.
- `save_model` and `load_model` record `n_layers`, but `load_model` does not pass it back to `OutbreakSTGNN`, so loaded models use that class's default (2 layers).
- `requirements.txt` needs cleanup (see Installation).

## License

No license file is currently included. Add one (for example MIT) if you want others to be able to reuse the code.
