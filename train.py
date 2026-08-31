import math
import numpy as np
import random
import torch
import torch.nn.functional as F
from scipy.signal import convolve2d
import gc

from data import (
    genReports,
    reports_to_sightings,
    generate_latlon_matrix,
    build_connections,
    load_random_ndvi,
    build_episode_tensors,
    N_FEATURES,
)
from model import OutbreakSTGNN, GridGeoref


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
            (math.floor(rando[1] / 3)), 2 * (math.floor(rando[1] / 3)) - 1
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
    Function to randomly choose beta and gamma for disease simulation

    inputs
    --------
    choice (optional) = override the random selection of disease spreadability
        options: "unviable", "slow", "normal", "fast", or "none" for random

    returns
    -------
    beta = variable for the spread rate of the disease
    gamma = rate the infected individuals are removed from the infected category
    """
    # Tiered sampling to get an even mix of disease types
    if choice in ["unviable", "slow", "normal", "fast"]:
        disease_type = choice
    else:
        disease_type = random.choice(["unviable", "slow", "normal", "fast"])
        disease_type = "fast"

    while True:
        if disease_type == "unviable":
            # Disease spreads quickly but dies out too fast to cause outbreak
            # High beta for fast spread, but very high gamma for quick recovery
            beta = random.uniform(0.30, 0.60)
            gamma = random.uniform(0.30, 0.60)
            # Beta/gamma ratio ~ 0.5-2.0 = disease can't sustain itself

        elif disease_type == "slow":
            # Slow spread with slow death/recovery
            # Low beta and low gamma = persistent but slow-moving disease
            beta = random.uniform(0.05, 0.15)
            gamma = random.uniform(0.01, 0.04)
            # Beta/gamma ratio ~ 2-10 = slow but sustainable spread

        elif disease_type == "normal":
            # Normal spread with slow death/recovery
            # Moderate beta with low gamma = steady, sustained outbreak
            beta = random.uniform(0.15, 0.35)
            gamma = random.uniform(0.02, 0.08)
            # Beta/gamma ratio ~ 3-15 = good sustained spread

        else:  # fast
            # Fast spread with normal death/recovery
            # High beta with moderate gamma = rapid, visible outbreak
            beta = random.uniform(0.35, 0.70)
            gamma = random.uniform(0.08, 0.20)
            # Beta/gamma ratio ~ 2-8 = fast but not instantly saturating

        # Ensure the parameters match the intended disease type
        ratio = beta / gamma

        if disease_type == "unviable":
            # Must be unviable (ratio < 2.0)
            if ratio < 2.0:
                return beta, gamma
        elif disease_type == "slow":
            # Slow spread with slow death (ratio 2-10)
            if 2.0 < ratio < 10.0:
                return beta, gamma
        elif disease_type == "normal":
            # Normal spread with slow death (ratio 3-15)
            if 3.0 < ratio < 15.0:
                return beta, gamma
        else:  # fast
            # Fast spread with normal death (ratio 2-8)
            if 2.0 < ratio < 8.0:
                return beta, gamma


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
    sigma = random.uniform(1, 4)
    wind_x = random.uniform(-4, 4)
    wind_y = random.uniform(-4, 4)

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

    returns
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
    # print(f"beta = {beta}")
    # print(f"gamma = {gamma}")
    conv_matrix = random_wind_kernel()
    S, I, R = start_SIR_random_middle(ndvi)

    S, I, R = sir_spatial_step(S, I, R, ndvi, conv_matrix, beta, gamma)

    s_hist = [S.copy()]
    i_hist = [I.copy()]

    for t in range(timesteps - 1):
        S, I, R = sir_spatial_step(S, I, R, ndvi, conv_matrix, beta, gamma)
        s_hist.append(S.copy())
        i_hist.append(I.copy())

    s_hist = np.stack(s_hist, axis=0)  # (T, n_rows, n_cols)
    i_hist = np.stack(i_hist, axis=0)  # (T, n_rows, n_cols)
    return s_hist, i_hist


def make_episode(
    ndvi_folder,
    used,
    dist_per_cell_meters=30,
    timesteps=50,
    fp=0.01,
    fn=0.01,
    plantspergrid=10,
    infected_threshold=0.1,
):
    """
    function to create a single training episode from start to finish

    loads a random ndvi tile, runs a simulated outbreak on it,
    generates daily sighting reports, and converts everything
    into the tensor format the model expects

    inputs
    --------
    ndvi_folder = name of the folder containing ndvi data
        passed directly to load_random_ndvi
    used = running Nx2 array of (lat, lon) pairs already used
        updated and returned so tiles are not repeated across calls within
        a training run
    dist_per_cell_meters = real-world size of one ndvi pixel in meters
        not stored in the png files themselves so must be provided manually
    timesteps = total number of days to simulate
    fp = false positive rate for sighting reports
    fn = false negative rate for sighting reports
    plantspergrid = number of plants per grid cell used for report generation
    infected_threshold = infected fraction above which a cell is considered
        "truly infected" for ground truth labels ground truth uses same-day
        infection state (target at day t = infected state at day t) not future
        prediction

    returns
    -------
    x_seq = (T, n_nodes, N_FEATURES) float32 tensor of input features
    y_seq = (T, n_nodes) float32 tensor of ground truth labels
    georef = GridGeoref object for coordinate conversions
    used = updated Nx2 array of (lat, lon) pairs with new tile added at top
    """
    ndvi, used = load_random_ndvi(ndvi_folder, used)
    n_rows, n_cols = ndvi.shape

    center_coords = tuple(used[0])

    locs = generate_latlon_matrix(
        ndvi, center_coords, dist_per_cell=dist_per_cell_meters
    )
    georef = GridGeoref(
        center_coords=center_coords,
        dist_per_cell=dist_per_cell_meters,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    ndvi_scaled = (0.5 + ndvi * 1.5) ** 2

    s_hist, i_hist = randomSimulation(ndvi_scaled, timesteps=timesteps)

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

    ground_truth_masks = (i_hist > infected_threshold).astype(np.float32)

    x_seq, y_seq = build_episode_tensors(
        ndvi, sightings, np.array(timestamps), georef, ground_truth_masks
    )
    return x_seq, y_seq, georef, used


def focal_loss_with_logits(logits, targets, gamma=2.0, alpha=0.25):
    """
    focal loss that down-weights easy examples and focuses on hard ones
    particularly good for imbalanced datasets

    inputs
    --------
    logits = model output logits
    targets = ground truth labels (0 or 1)
    gamma = focusing parameter (higher = more focus on hard examples)
    alpha = weight for positive class

    returns
    -------
    loss = focal loss value
    """
    BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-BCE_loss)
    alpha_weight = alpha * targets + (1 - alpha) * (1 - targets)
    focal_loss = alpha_weight * (1 - pt) ** gamma * BCE_loss
    return focal_loss.mean()


def dice_loss_with_logits(logits, targets, smooth=1.0):
    """
    Dice loss for spatial prediction tasks

    inputs
    --------
    logits = model output logits (before sigmoid)
    targets = ground truth labels (0 or 1)
    smooth = smoothing factor to prevent division by zero

    returns
    -------
    loss = dice loss value
    """
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)

    intersection = (probs_flat * targets_flat).sum()
    union = probs_flat.sum() + targets_flat.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice


def combined_loss(logits, targets, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
    """
    Combine BCE and Dice loss for better spatial learning
    """
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets)

    dice = dice_loss_with_logits(logits, targets)
    return bce_weight * bce + dice_weight * dice


def calculate_pos_weight(y_seq, max_weight=100.0):
    """
    calculate positive class weight based on actual frequency
    """
    num_pos = y_seq.sum()
    num_neg = y_seq.numel() - num_pos
    if num_pos == 0:
        return torch.tensor([max_weight]).to(y_seq.device)
    pos_weight = num_neg / num_pos
    pos_weight = torch.clamp(pos_weight, 1.0, max_weight)
    return pos_weight


def tversky_loss(logits, targets, alpha=0.7, beta=0.3, smooth=1.0):
    """
    Tversky loss - asymmetric penalty for imbalanced data

    inputs
    --------
    logits = model output logits
    targets = ground truth labels (0 or 1)
    alpha = weight for false negatives (missed infections)
            HIGHER = penalize missing infections more
    beta = weight for false positives (false alarms)
           LOWER = penalize false alarms less
    smooth = smoothing factor to prevent division by zero

    returns
    -------
    loss = tversky loss value (1 - tversky index)
    """
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)

    tp = (probs_flat * targets_flat).sum()
    fn = ((1 - probs_flat) * targets_flat).sum()
    fp = (probs_flat * (1 - targets_flat)).sum()

    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - tversky


def train(
    ndvi_folder,
    model=None,
    n_epochs=20,
    episodes_per_epoch=10,
    timesteps=50,
    lr=1e-3,
    hidden_channels=256,
    dist_per_cell_meters=30,
    device="cpu",
    loss_type="tversky",
    tversky_alpha=0.8,
    tversky_beta=0.2,
    optimizer_type="adam",
    chunk_size=10,  # Process 10 timesteps at a time
):
    """
    Training with chunked processing to save memory
    """
    if model is None:
        model = OutbreakSTGNN(
            in_channels=N_FEATURES, hidden_channels=hidden_channels
        ).to(device)
    else:
        # Use existing model
        model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    used = np.empty((0, 2))

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_done = 0

        for episode_idx in range(episodes_per_epoch):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            model.train()
            optimizer.zero_grad()

            x_seq, y_seq, georef, used = make_episode(
                ndvi_folder,
                used,
                dist_per_cell_meters=dist_per_cell_meters,
                timesteps=timesteps,
            )

            edge_index, edge_weight = build_connections(georef.n_rows, georef.n_cols)
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)

            # Process in chunks
            total_loss = 0.0
            n_chunks = 0

            for start_idx in range(0, timesteps, chunk_size):
                end_idx = min(start_idx + chunk_size, timesteps)

                # Move chunk to device
                x_chunk = x_seq[start_idx:end_idx].to(device)
                y_chunk = y_seq[start_idx:end_idx].to(device)

                # Forward pass on chunk
                logits_list = model(x_chunk, edge_index, edge_weight)

                # Calculate loss on chunk
                chunk_loss = 0.0
                for t, logits in enumerate(logits_list):
                    chunk_loss = chunk_loss + tversky_loss(
                        logits, y_chunk[t], alpha=tversky_alpha, beta=tversky_beta
                    )
                chunk_loss = chunk_loss / len(logits_list)

                # Backward pass
                chunk_loss.backward()

                total_loss += chunk_loss.item()
                n_chunks += 1

                # Free chunk memory
                del x_chunk, y_chunk, logits_list, chunk_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Average loss over chunks
            avg_loss = total_loss / n_chunks

            # Update weights after all chunks
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += avg_loss
            n_done += 1

            # Free episode memory
            del x_seq, y_seq, edge_index, edge_weight
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        print(
            f"epoch {epoch + 1}/{n_epochs}  avg loss={epoch_loss / max(n_done, 1):.4f}"
        )

    return model


def save_model(
    model,
    optimizer,
    epoch,
    loss,
    filepath="outbreak_model.pth",
    hidden_channels=256,
    n_layers=3,
    loss_type="tversky",
    tversky_alpha=0.8,
    tversky_beta=0.2,
    optimizer_type="adam",
    lr=1e-3,
    additional_info=None,
):
    """
    function to save the model and training state

    inputs
    --------
    model = trained OutbreakSTGNN model
    optimizer = optimizer used during training
    epoch = current epoch number
    loss = current loss value
    filepath = location to save the model checkpoint
    hidden_channels = hidden channel size used in the model
    n_layers = number of DCRNN layers used
    loss_type = type of loss function used
    tversky_alpha = alpha parameter for Tversky loss
    tversky_beta = beta parameter for Tversky loss
    optimizer_type = type of optimizer used
    lr = learning rate used
    additional_info = any extra information to save (dict)

    returns
    -------
    none, saves model to disk
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "model_config": {
            "in_channels": N_FEATURES,
            "hidden_channels": hidden_channels,
            "n_layers": n_layers,
        },
        "training_info": {
            "loss_type": loss_type,
            "tversky_alpha": tversky_alpha,
            "tversky_beta": tversky_beta,
            "optimizer": optimizer_type,
            "lr": lr,
        },
    }

    # Add any additional info
    if additional_info:
        checkpoint.update(additional_info)

    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")
    print(f"  Epoch: {epoch}")
    print(f"  Loss: {loss:.4f}")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Layers: {n_layers}")


def load_model(filepath="outbreak_model.pth", device="cpu", load_optimizer=False):
    """
    function to load a saved model checkpoint

    inputs
    --------
    filepath = location of the saved model
    device = device to load the model onto ("cpu" or "cuda")
    load_optimizer = whether to also load optimizer state
                    (True for continuing training, False for inference)

    returns
    -------
    model = loaded OutbreakSTGNN model
    optimizer = loaded optimizer (if load_optimizer=True, else None)
    checkpoint = full checkpoint dictionary with training info
    """
    checkpoint = torch.load(filepath, map_location=device)

    # Extract model configuration
    model_config = checkpoint.get("model_config", {})
    in_channels = model_config.get("in_channels", N_FEATURES)
    hidden_channels = model_config.get("hidden_channels", 256)
    n_layers = model_config.get("n_layers", 3)

    # Recreate model architecture
    model = OutbreakSTGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set to evaluation mode by default
    model.eval()

    # Optionally load optimizer
    optimizer = None
    if load_optimizer and "optimizer_state_dict" in checkpoint:
        training_info = checkpoint.get("training_info", {})
        optimizer_type = training_info.get("optimizer", "adam")
        lr = training_info.get("lr", 1e-3)

        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Print loading info
    print(f"Model loaded from {filepath}")
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if "loss" in checkpoint:
        print(f"  Loss: {checkpoint['loss']:.4f}")
    if "model_config" in checkpoint:
        print(f"  Hidden channels: {model_config.get('hidden_channels', 'unknown')}")
        print(f"  Layers: {model_config.get('n_layers', 'unknown')}")
    if "training_info" in checkpoint:
        info = checkpoint["training_info"]
        print(f"  Loss type: {info.get('loss_type', 'unknown')}")
        print(f"  Optimizer: {info.get('optimizer', 'unknown')}")

    return model, optimizer, checkpoint


def save_model_checkpoint(model, optimizer, epoch, loss, filepath, **kwargs):
    """
    simplified save function that accepts any additional info as keyword arguments

    inputs
    --------
    model = trained model
    optimizer = optimizer state
    epoch = current epoch
    loss = current loss
    filepath = save location
    **kwargs = any additional information to save

    returns
    -------
    none
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "model_config": {
            "in_channels": N_FEATURES,
            "hidden_channels": kwargs.get("hidden_channels", 256),
            "n_layers": kwargs.get("n_layers", 3),
        },
        "training_info": {
            "loss_type": kwargs.get("loss_type", "tversky"),
            "tversky_alpha": kwargs.get("tversky_alpha", 0.8),
            "tversky_beta": kwargs.get("tversky_beta", 0.2),
            "optimizer": kwargs.get("optimizer_type", "adam"),
            "lr": kwargs.get("lr", 1e-3),
        },
    }

    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def test_overfitting(device="cuda"):
    """Test if model can overfit to a single example"""
    model = OutbreakSTGNN(in_channels=N_FEATURES, hidden_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Generate one episode
    x_seq, y_seq, georef, used = make_episode(
        "brazilFarms",
        np.empty((0, 2)),
        dist_per_cell_meters=30,
        timesteps=20,
    )

    # Move ALL tensors to device
    x_seq = x_seq.to(device)
    y_seq = y_seq.to(device)
    edge_index, edge_weight = build_connections(georef.n_rows, georef.n_cols)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # Train on same example
    for i in range(200):
        optimizer.zero_grad()
        logits_list = model(x_seq, edge_index, edge_weight)
        loss = sum(
            [
                F.binary_cross_entropy_with_logits(logits, y_seq[t])
                for t, logits in enumerate(logits_list)
            ]
        ) / len(logits_list)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Iteration {i}: loss={loss.item():.6f}")

    return loss.item()


def test_overfitting_larger(device="cuda", n_iterations=300):
    """Test with larger hidden channels"""
    # Try different hidden sizes
    for hidden_size in [64, 128, 256]:
        print(f"\nTesting with hidden_channels={hidden_size}")

        model = OutbreakSTGNN(in_channels=N_FEATURES, hidden_channels=hidden_size).to(
            device
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-2
        )  # Higher LR for larger model

        # Generate one episode
        x_seq, y_seq, georef, used = make_episode(
            "brazilFarms",
            np.empty((0, 2)),
            dist_per_cell_meters=30,
            timesteps=20,
        )

        x_seq = x_seq.to(device)
        y_seq = y_seq.to(device)
        edge_index, edge_weight = build_connections(georef.n_rows, georef.n_cols)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

        # Train
        for i in range(n_iterations):
            optimizer.zero_grad()
            logits_list = model(x_seq, edge_index, edge_weight)
            loss = sum(
                [
                    F.binary_cross_entropy_with_logits(logits, y_seq[t])
                    for t, logits in enumerate(logits_list)
                ]
            ) / len(logits_list)
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"  Iteration {i}: loss={loss.item():.6f}")

        print(f"  Final loss: {loss.item():.6f}")
        if loss.item() < 0.001:
            print(f"  ✓ Can overfit with {hidden_size} hidden channels!")
            break
