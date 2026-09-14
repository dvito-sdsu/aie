"""
Spatial SIR-style plant disease simulation.

The model stores susceptible (S), infected (I), and recovered/removed (R)
proportions for each grid cell. Infection spreads between cells via a 2D
Gaussian kernel, and observations/reports can include false positives and
false negatives.
"""

import math
import random

import numpy as np
from scipy.signal import convolve2d


class simModel:
    """
    Spatial SIR-like simulator over an NDVI grid.

    Parameters
    ----------
    ndvi : array-like
        Per-cell NDVI values. Higher values generally increase infection pressure.
    center_coords : tuple
        (latitude, longitude) of the grid center.
    dist_per_grid : float
        Grid-cell size in meters.
    fp : float
        False-positive probability for reports.
    fn : float
        False-negative probability for reports.
    beta : float or None
        Infection/transmission rate. Must be supplied together with gamma.
    gamma : float or None
        Recovery/removal rate. Must be supplied together with beta.
    """

    def __init__(
        self,
        ndvi,
        center_coords,
        dist_per_grid=15,
        fp=0.01,
        fn=0.01,
        beta=None,
        gamma=None,
    ):
        self.ndvi = ndvi
        self.coords = center_coords
        self.dist = dist_per_grid
        self.fp = fp
        self.fn = fn

        # beta and gamma must be provided together. If exactly one is None,
        # the XOR expression is True and the `not` makes the condition False.
        # If both are None or both are provided, the parameters are accepted.
        if not ((beta == None) ^ (gamma == None)):
            self.beta = beta
            self.gamma = gamma
        else:
            raise ValueError(
                "Only one beta or gamma was provided, requires neither or both"
            )

    def createMatrix(self, radius=2, sigma=1.0, wind_x=0.0, wind_y=0.0):
        """
        Create and store a normalized 2D Gaussian dispersal kernel.

        radius : int
            Kernel radius. The full kernel size is (2 * radius + 1) by
            (2 * radius + 1). For example, radius=2 gives a 5x5 kernel.
        sigma : float
            Standard deviation of the Gaussian; larger values spread infection
            farther in one step.
        wind_x, wind_y : float
            Offsets that shift the Gaussian peak away from the center. This can
            represent wind-biased spread.
        """
        # radius of 2 => 5x5 convolution matrix
        size = 2 * radius + 1

        # Create a coordinate grid centered at (0, 0).
        # For size=5, ax = [-2, -1, 0, 1, 2].
        ax = np.linspace(-(size // 2), size // 2, size)
        xx, yy = np.meshgrid(ax, ax)

        # Evaluate an unnormalized Gaussian. Subtracting wind_x/wind_y shifts
        # the peak of the kernel in the x/y directions.
        kernel = np.exp(-((xx - wind_x) ** 2 + (yy - wind_y) ** 2) / (2 * sigma**2))

        # Normalize so the kernel sums to 1. This keeps convolution output on
        # a comparable scale to the input infected density.
        self.kernel = kernel / np.sum(kernel)

    def randomBG(self, choice=None):
        """
        Randomly set beta and gamma according to a disease archetype.

        Archetypes:
          - "unviable": spreads quickly but dies out quickly.
          - "slow": low transmission and low recovery; persistent but slow.
          - "normal": moderate transmission and low recovery; sustained spread.
          - "fast": high transmission and moderate recovery; rapid outbreak.

        If `choice` is invalid or None, the current implementation randomly
        chooses an archetype and then immediately overwrites it with "fast".
        """
        if choice in ["unviable", "slow", "normal", "fast"]:
            disease_type = choice
        else:
            disease_type = random.choice(["unviable", "slow", "normal", "fast"])
            # NOTE: This line overrides the random choice above, so the default
            # behavior is always "fast" when `choice` is not explicitly valid.
            disease_type = "fast"

        while True:
            if disease_type == "unviable":
                # Disease spreads quickly but dies out too fast to cause outbreak.
                # High beta for fast spread, but very high gamma for quick recovery.
                beta = random.uniform(0.30, 0.60)
                gamma = random.uniform(0.30, 0.60)
                # Beta/gamma ratio ~ 0.5-2.0 = disease can't sustain itself.

            elif disease_type == "slow":
                # Slow spread with slow death/recovery.
                # Low beta and low gamma = persistent but slow-moving disease.
                beta = random.uniform(0.05, 0.15)
                gamma = random.uniform(0.01, 0.04)
                # Beta/gamma ratio ~ 2-10 = slow but sustainable spread.

            elif disease_type == "normal":
                # Normal spread with slow death/recovery.
                # Moderate beta with low gamma = steady, sustained outbreak.
                beta = random.uniform(0.15, 0.35)
                gamma = random.uniform(0.02, 0.08)
                # Beta/gamma ratio ~ 3-15 = good sustained spread.

            else:  # fast
                # Fast spread with normal death/recovery.
                # High beta with moderate gamma = rapid, visible outbreak.
                beta = random.uniform(0.35, 0.70)
                gamma = random.uniform(0.08, 0.20)
                # Beta/gamma ratio ~ 2-8 = fast but not instantly saturating.

            # Ensure the sampled parameters match the intended disease type.
            ratio = beta / gamma

            if disease_type == "unviable":
                # Must be unviable (ratio < 2.0).
                if ratio < 2.0:
                    break
            elif disease_type == "slow":
                # Slow spread with slow death (ratio 2-10).
                if 2.0 < ratio < 10.0:
                    break
            elif disease_type == "normal":
                # Normal spread with slow death (ratio 3-15).
                if 3.0 < ratio < 15.0:
                    break
            else:  # fast
                # Fast spread with normal death (ratio 2-8).
                if 2.0 < ratio < 8.0:
                    break

        self.beta = beta
        self.gamma = gamma

    def setBG(self, beta, gamma):
        """Manually set the transmission and recovery rates."""
        self.beta = beta
        self.gamma = gamma

    def randomMatrix(self):
        """Create a randomized Gaussian dispersal kernel and store it."""
        radius = random.randint(3, 7)
        sigma = random.uniform(1, 4)
        wind_x = random.uniform(-4, 4)
        wind_y = random.uniform(-4, 4)

        self.createMatrix(radius=radius, sigma=sigma, wind_x=wind_x, wind_y=wind_y)

    def initSim(self, mode="middle"):
        """
        Initialize the SIR arrays and set the simulation day to 0.

        mode="middle" places the initial infection near the middle of the grid.
        mode="random" is intended to place it randomly, but the `randomStart`
        method is not defined in this file.
        """
        if mode not in ("middle", "random"):
            raise ValueError("Mode not recognized, must be either middle or random")

        if mode == "middle":
            self.randomMiddle()
        else:
            # NOTE: randomStart() is referenced but not defined in this file.
            self.randomStart()

        self.day = 0

    def randomMiddle(self):
        """
        Initialize one infected cell in the middle third of the grid.

        Sets:
          self.S, self.I, self.R : current SIR arrays
          self.S_hist, self.I_hist : history arrays for S and I
          self.coordinate_matrix : per-cell latitude/longitude lookup
        """
        # Convert input to a numpy array to safely extract its shape.
        input_np = np.asarray(self.ndvi)

        # 1. Create the infected array: all zeros, then set one cell to 1.
        I = np.zeros_like(input_np)
        rando = I.shape

        # Pick a random row and column within the middle third of the grid.
        # The bounds below select rows/cols from floor(n/3) to 2*floor(n/3)-1.
        I[
            np.random.randint(
                (math.floor(rando[0] / 3)), 2 * (math.floor(rando[0] / 3)) - 1
            )
        ][
            np.random.randint(
                (math.floor(rando[1] / 3)), 2 * (math.floor(rando[1] / 3)) - 1
            )
        ] = 1

        # 2. Create susceptible and recovered arrays.
        # S is 1 everywhere except the initially infected cell, which becomes 0.
        S = np.zeros_like(input_np) + 1 - I
        R = np.zeros_like(input_np)

        # Save the initial timestep in the object.
        self.S = S
        self.I = I
        self.R = R

        # Build the latitude/longitude lookup for every grid cell.
        self.generate_latlon_matrix()

        # Save initial history. Note: R history is not tracked here.
        self.S_hist = S
        self.I_hist = I

    def step(self, steps=1):
        """
        Advance the simulation by one or more time steps.

        Each step:
          1. Convolve infected cells with the dispersal kernel to get infection pressure.
          2. Compute new infections from beta, S, NDVI, and infection pressure.
          3. Compute recoveries from gamma and I.
          4. Update S, I, and R, clipping values to [0, 1].
          5. Append history and generate reports for the current day.
        """
        if steps < 1:
            raise AttributeError(
                f"Step was called with {steps} steps, must call with 1 or more"
            )

        for i in range(steps):
            # Infection pressure at each cell: weighted sum of nearby infected cells.
            # mode="same" returns the same shape; boundary="fill" treats outside
            # the grid as zero.
            infected_pressure = convolve2d(
                self.I, self.kernel, mode="same", boundary="fill"
            )

            # New infections depend on transmission rate, susceptible density,
            # NDVI (plant density/vulnerability proxy), and infection pressure.
            new_infections = self.beta * self.S * self.ndvi * infected_pressure

            # Recovery/removal is proportional to the infected density.
            new_recoveries = self.gamma * self.I

            # Do not infect or recover more than the available amount in a cell.
            # This also prevents negative values in the next step.
            new_infections = np.minimum(new_infections, self.S)
            new_recoveries = np.minimum(new_recoveries, self.I)

            # Compute the next SIR state.
            S_next = self.S - new_infections
            I_next = self.I + new_infections - new_recoveries
            R_next = self.R + new_recoveries

            # Keep each compartment within [0, 1].
            # NOTE: clipping each compartment independently can slightly break
            # the exact S + I + R = 1 conservation.
            self.S = np.clip(S_next, 0.0, 1.0)
            self.I = np.clip(I_next, 0.0, 1.0)
            self.R = np.clip(R_next, 0.0, 1.0)

            # Append the new state to history.
            self.S_hist = np.vstack((self.S_hist, self.S.copy()))
            self.I_hist = np.vstack((self.I_hist, self.I.copy()))

            # Update reports. On the first call, self.reports does not exist,
            # so the AttributeError branch initializes it.
            try:
                self.day = self.day + 1
                self.reports = np.vstack((self.reports, self.createReports()))
            except AttributeError:
                self.day = 1
                self.reports = self.createReports()

    def getBG(self):
        """Print and return the current beta and gamma values."""
        print(f"Beta is: {self.beta}")
        print(f"Gamma is: {self.gamma}")
        return self.beta, self.gamma

    def createReports(self, seed=None):
        """
        Generate observation reports for the current day.

        Returns
        -------
        reports : ndarray, shape (M, 3)
            Each row is [latitude, longitude, day]. Returns an empty (0, 3)
            array if no reports are generated.

        Current assumptions:
          - The entire area is scanned by the disease-detection model.
          - True positives are drawn as Poisson counts from I * NDVI.
          - False negatives are applied with probability self.fn.
          - False positives are drawn as Poisson counts from S * NDVI * self.fp.
        """
        # Possible fixes/considerations:
        # - Restrict reports to be within a certain distance of the outbreak;
        #   this may be computationally intensive.
        # - Make the NDVI range smaller so reports cover fewer farms at once.
        #   Some large farms may not be fully covered, while multiple small
        #   farms can be covered in one cell.

        rng = np.random.default_rng(seed)
        plantidx = self.ndvi

        # True positive reports: Poisson-distributed counts based on infected
        # density and NDVI.
        counts = rng.poisson(self.I * plantidx)
        rows, cols = np.nonzero(counts)
        n = counts[rows, cols]

        # Expand each cell into repeated rows/cols according to its count.
        rows_rep = np.repeat(rows, n)
        cols_rep = np.repeat(cols, n)

        # False negatives: each true positive has self.fn chance of being missed.
        if len(rows_rep) > 0:
            keep = rng.random(len(rows_rep)) >= self.fn
            rows_rep = rows_rep[keep]
            cols_rep = cols_rep[keep]

        # False positives: susceptible cells can generate spurious reports.
        fp_counts = rng.poisson(self.S * plantidx * self.fp)
        fp_rows, fp_cols = np.nonzero(fp_counts)
        fp_n = fp_counts[fp_rows, fp_cols]
        fp_rows_rep = np.repeat(fp_rows, fp_n)
        fp_cols_rep = np.repeat(fp_cols, fp_n)

        # Combine true positives and false positives.
        all_rows = np.concatenate([rows_rep, fp_rows_rep])
        all_cols = np.concatenate([cols_rep, fp_cols_rep])

        if len(all_rows) == 0:
            return np.empty((0, 3), dtype=float)

        # Shuffle so true-positive and false-positive reports are interleaved
        # rather than false positives always trailing at the end.
        perm = rng.permutation(len(all_rows))
        all_rows = all_rows[perm]
        all_cols = all_cols[perm]

        # self.coordinate_matrix is (rows, cols, 2) with the last axis ordered
        # as [latitude, longitude].
        lat = self.coordinate_matrix[all_rows, all_cols, 0].astype(float)
        lon = self.coordinate_matrix[all_rows, all_cols, 1].astype(float)

        # Jitter each report to a random point inside its grid cell rather than
        # always reporting the exact cell-center coordinate.
        # NOTE: self.dist is in meters, but lat/lon are in degrees. This jitter
        # likely needs a meters-to-degrees conversion, similar to the scales
        # computed in generate_latlon_matrix().
        lat = lat + rng.uniform(-self.dist / 2, self.dist / 2, size=lat.shape)
        lon = lon + rng.uniform(-self.dist / 2, self.dist / 2, size=lon.shape)

        day_col = np.full(lat.shape, self.day, dtype=float)

        reports = np.column_stack([lat, lon, day_col])

        return reports

    def generate_latlon_matrix(self):
        """
        Build a per-cell latitude/longitude lookup matrix.

        The grid is centered on self.coords. Rows move north/south and columns
        move east/west. Distances are converted from meters to degrees using
        an equirectangular approximation around the center latitude.
        """
        rows, cols = self.I.shape
        center_row, center_col = rows // 2, cols // 2
        center_lat, center_lon = self.coords

        # Constants.
        R = 6378137.0  # Earth radius in meters.
        RAD_PER_DEG = np.pi / 180.0
        DEG_PER_RAD = 180.0 / np.pi

        # Longitude degrees are smaller near the poles, so scale by cos(latitude).
        cos_lat = np.cos(center_lat * RAD_PER_DEG)
        lat_scale = (self.dist / R) * DEG_PER_RAD
        lon_scale = (self.dist / (R * cos_lat)) * DEG_PER_RAD

        # Create index offsets relative to the grid center.
        # y_indices increases upward (north), so row 0 is north of the center.
        y_indices = -(np.arange(rows) - center_row)
        x_indices = np.arange(cols) - center_col

        x_offsets, y_offsets = np.meshgrid(x_indices, y_indices)

        # Convert offsets to latitude/longitude offsets.
        target_lats = center_lat + (y_offsets * lat_scale)
        target_lons = center_lon + (x_offsets * lon_scale)

        # Stack into an (rows, cols, 2) array: last axis is [lat, lon].
        coordinate_matrix = np.dstack((target_lats, target_lons))

        self.coordinate_matrix = coordinate_matrix
