import os
import numpy as np


# ==========================================================
# 8-STATE TERRAIN CELLULAR AUTOMATA
# ==========================================================

EMPTY = 0
WATER = 1
SEDIMENT = 2
SOIL = 3
SLOPE = 4
ROCK = 5
PLATEAU = 6
PEAK = 7


STATE_NAMES = {
    EMPTY: "EMPTY",
    WATER: "WATER",
    SEDIMENT: "SEDIMENT",
    SOIL: "SOIL",
    SLOPE: "SLOPE",
    ROCK: "ROCK",
    PLATEAU: "PLATEAU",
    PEAK: "PEAK"
}


# ==========================================================
# STATE CLASSIFIER
# ==========================================================

class TerrainStateMapper:

    def __init__(self, height_grid):

        self.height = height_grid.astype(np.float32)

        self.rows = self.height.shape[0]
        self.cols = self.height.shape[1]

        self.state_grid = np.zeros_like(
            self.height,
            dtype=np.uint8
        )

    # ======================================================
    # Calculate Local Slope
    # ======================================================

    def calculate_slope(self):

        gy, gx = np.gradient(self.height)

        slope = np.sqrt(

            gx * gx +

            gy * gy

        )

        return slope
    # ======================================================
    # Create Adaptive Terrain States
    # ======================================================

    def create_states(self):

        height = self.height

        slope = self.calculate_slope()

        # --------------------------------------------------
        # Reset state grid
        # --------------------------------------------------

        self.state_grid[:] = EMPTY

        # --------------------------------------------------
        # Terrain mask
        # --------------------------------------------------

        terrain_mask = height > 0

        if not np.any(terrain_mask):

            return self.state_grid

        # --------------------------------------------------
        # Base height
        # --------------------------------------------------

        base_height = np.min(
            height[terrain_mask]
        )

        # --------------------------------------------------
        # Elevated terrain
        # --------------------------------------------------

        elevated_mask = (
            terrain_mask &
            (height > base_height)
        )

        if not np.any(elevated_mask):

            self.state_grid[terrain_mask] = SOIL

            return self.state_grid

        # --------------------------------------------------
        # Height distribution
        # --------------------------------------------------

        elevated_heights = height[elevated_mask]

        h50 = np.percentile(
            elevated_heights,
            50
        )

        h75 = np.percentile(
            elevated_heights,
            75
        )

        h90 = np.percentile(
            elevated_heights,
            90
        )

        h97 = np.percentile(
            elevated_heights,
            97
        )

        # --------------------------------------------------
        # Slope distribution
        # --------------------------------------------------

        elevated_slopes = slope[elevated_mask]

        s50 = np.percentile(
            elevated_slopes,
            50
        )

        s70 = np.percentile(
            elevated_slopes,
            70
        )

        # --------------------------------------------------
        # SOIL
        # --------------------------------------------------

        soil_mask = (
            terrain_mask &
            (height <= h50)
        )

        self.state_grid[soil_mask] = SOIL

        # --------------------------------------------------
        # SLOPE
        # --------------------------------------------------

        slope_mask = (
            elevated_mask &
            (height > h50) &
            (height < h75) &
            (slope >= s50)
        )

        self.state_grid[slope_mask] = SLOPE

        # --------------------------------------------------
        # ROCK
        # --------------------------------------------------

        rock_mask = (
            elevated_mask &
            (
                (
                    (height >= h75) &
                    (slope >= s70)
                )
                |
                (
                    (height >= h90) &
                    (slope >= s50)
                )
            )
        )

        self.state_grid[rock_mask] = ROCK

        # --------------------------------------------------
        # PLATEAU
        # High + relatively flat
        # --------------------------------------------------

        plateau_mask = (
            elevated_mask &
            (height >= h75) &
            (slope < s50)
        )

        self.state_grid[plateau_mask] = PLATEAU

        # --------------------------------------------------
        # PEAK
        # Highest terrain
        # --------------------------------------------------

        peak_mask = (
            elevated_mask &
            (height >= h97)
        )

        self.state_grid[peak_mask] = PEAK

        # --------------------------------------------------
        # Safety
        # Any remaining elevated cells become SLOPE
        # --------------------------------------------------

        remaining = (
            elevated_mask &
            (self.state_grid == EMPTY)
        )

        self.state_grid[remaining] = SLOPE

        return self.state_grid

    # ======================================================
    # Statistics
    # ======================================================

    def statistics(self):

        print("\n" + "=" * 60)
        print("CA STATE STATISTICS")
        print("=" * 60)

        for state in range(8):

            count = np.sum(

                self.state_grid == state

            )

            print(

                f"{state} - "

                f"{STATE_NAMES[state]:10s} : "

                f"{count}"

            )

    # ======================================================
    # Save State Grid
    # ======================================================

    def save(self, filename="state_grid.npy"):

        os.makedirs(

            "OUTPUT",

            exist_ok=True

        )

        path = os.path.join(

            "OUTPUT",

            filename

        )

        np.save(

            path,

            self.state_grid

        )

        print(

            "\nState Grid Saved:",

            path

        )


# ==========================================================
# TEST / STANDALONE EXECUTION
# ==========================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("TERRAIN CA STATE MAPPER")
    print("=" * 60)

    height_path = os.path.join(

        "OUTPUT",

        "height_grid.npy"

    )

    if not os.path.exists(height_path):

        raise FileNotFoundError(

            "OUTPUT/height_grid.npy not found. "

            "Run placement_engine.py first."

        )

    height_grid = np.load(

        height_path

    )

    print(

        "\nHeight Grid:",

        height_grid.shape

    )

    print(

        "Maximum Height:",

        np.max(height_grid)

    )

    mapper = TerrainStateMapper(

        height_grid

    )

    state_grid = mapper.create_states()

    mapper.statistics()

    mapper.save()
