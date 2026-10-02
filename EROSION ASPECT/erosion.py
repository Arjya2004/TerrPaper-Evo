import os
import numpy as np


class TerrainErosion:

    # ==========================================================
    # Constructor
    # ==========================================================

    def __init__(

        self,

        height_grid,

        thermal_iterations=40,

        talus_angle=2.5,

        thermal_strength=0.20,

        max_move=0.25,

        peak_protection=True,

        peak_ratio=0.98

    ):

        # ------------------------------------------
        # Terrain
        # ------------------------------------------

        self.height = height_grid.astype(
            np.float32
        ).copy()

        self.rows, self.cols = self.height.shape

        # ------------------------------------------
        # Thermal Parameters
        # ------------------------------------------

        self.thermal_iterations = thermal_iterations

        self.talus_angle = talus_angle

        self.thermal_strength = thermal_strength

        self.max_move = max_move

        # ------------------------------------------
        # Peak Protection
        # ------------------------------------------

        self.peak_protection = peak_protection

        self.peak_ratio = peak_ratio

        # ------------------------------------------
        # Initial Statistics
        # ------------------------------------------

        self.initial_max = np.max(self.height)

        self.initial_mean = np.mean(self.height)

        self.initial_volume = np.sum(self.height)


    # ==========================================================
    # Check Boundary
    # ==========================================================

    def inside(

        self,

        r,

        c

    ):

        return (

            0 <= r < self.rows

            and

            0 <= c < self.cols

        )


    # ==========================================================
    # Get Moore Neighbours
    # ==========================================================

    def neighbours(

        self,

        r,

        c

    ):

        cells = []

        directions = [

            (-1,-1), (-1,0), (-1,1),

            (0,-1),          (0,1),

            (1,-1), (1,0), (1,1)

        ]

        for dr, dc in directions:

            nr = r + dr

            nc = c + dc

            if self.inside(

                nr,

                nc

            ):

                cells.append(

                    (nr, nc)

                )

        return cells


    # ==========================================================
    # Peak Limit
    # ==========================================================

    def peak_limit(self):

        if self.peak_protection:

            return (

                self.initial_max

                *

                self.peak_ratio

            )

        return float("inf")
           # ==========================================================
    # One Thermal Erosion Step
    # ==========================================================

    def thermal_step(self):

        delta = np.zeros_like(
            self.height,
            dtype=np.float32
        )

        active_cells = 0
        moved_material = 0.0

        peak = self.peak_limit()

        # ------------------------------------------------------
        # Visit every terrain cell
        # ------------------------------------------------------

        for r in range(1, self.rows - 1):

            for c in range(1, self.cols - 1):

                current = self.height[r, c]

                # Ignore empty cells
                if current <= 0:
                    continue

                # --------------------------------------------------
                # Soft Peak Protection
                # --------------------------------------------------

                if current >= peak:

                    peak_factor = 0.20

                elif current >= peak * 0.90:

                    peak_factor = 0.50

                else:

                    peak_factor = 1.00

                downhill = []
                                # --------------------------------------------------
                # Collect Downhill Neighbours
                # --------------------------------------------------

                for nr, nc in self.neighbours(r, c):

                    neighbour = self.height[nr, nc]

                    drop = current - neighbour

                    # Ignore stable neighbours
                    if drop <= self.talus_angle:
                        continue

                    # Distance Weight
                    if abs(nr - r) + abs(nc - c) == 1:
                        distance = 1.0          # N,S,E,W
                    else:
                        distance = 1.41421356   # Diagonal

                    weight = drop / distance

                    downhill.append(

                        (

                            nr,
                            nc,
                            drop,
                            weight

                        )

                    )

                # No unstable neighbours

                if len(downhill) == 0:
                    continue

                active_cells += 1

                # --------------------------------------------------
                # Sort by Weight
                # --------------------------------------------------

                downhill.sort(

                    key=lambda x: x[3],

                    reverse=True

                )

                # --------------------------------------------------
                # Soft Peak Soil Budget
                # --------------------------------------------------

                steepest_drop = downhill[0][2]

                soil_budget = (

                    (steepest_drop - self.talus_angle)

                    * self.thermal_strength

                    * peak_factor

                )

                soil_budget = min(

                    soil_budget,

                    self.max_move

                )

                if soil_budget <= 0:
                    continue
                                # --------------------------------------------------
                # Weighted Material Distribution
                # --------------------------------------------------

                total_weight = sum(

                    weight

                    for _, _, _, weight

                    in downhill

                )

                if total_weight <= 0:

                    continue

                # ----------------------------------------------
                # Move Material
                # ----------------------------------------------

                for nr, nc, drop, weight in downhill:

                    share = (

                        soil_budget

                        *

                        (weight / total_weight)

                    )

                    # Prevent unrealistic deposition

                    share = min(

                        share,

                        self.max_move * 0.50

                    )

                    delta[r, c] -= share

                    delta[nr, nc] += share

                    moved_material += share
                                    # --------------------------------------------------
                # Prevent Negative Height (Safety)
                # --------------------------------------------------

                delta[r, c] = max(

                    delta[r, c],

                    -current

                )

        # ----------------------------------------------------------
        # Return One Thermal Step
        # ----------------------------------------------------------

        return (

            delta,

            active_cells,

            moved_material

        )
        
            # ==========================================================
    # Thermal Erosion Simulation
    # ==========================================================

    def thermal_erosion(self):

        print("\n" + "=" * 60)
        print("THERMAL EROSION SIMULATION")
        print("=" * 60)

        previous_height = self.height.copy()

        for iteration in range(self.thermal_iterations):

            delta, active_cells, moved_material = self.thermal_step()

            # ------------------------------------------
            # Apply Delta
            # ------------------------------------------

            self.height += delta

            # Never allow negative height

            self.height = np.maximum(

                self.height,

                0.0

            )

            # ------------------------------------------
            # Statistics
            # ------------------------------------------

            max_change = np.max(

                np.abs(

                    self.height - previous_height

                )

            )

            previous_height = self.height.copy()

            peak = np.max(self.height)

            minimum = np.min(self.height)

            average = np.mean(self.height)

            volume = np.sum(self.height)

            print(

                f"Iteration {iteration+1:3d}"

                f" | Active={active_cells:5d}"

                f" | Moved={moved_material:8.3f}"

                f" | MaxΔ={max_change:7.4f}"

                f" | Peak={peak:7.2f}"

                f" | Avg={average:7.2f}"

                f" | Min={minimum:7.2f}"

                f" | Volume={volume:10.2f}"

            )

            # ------------------------------------------
            # Stop Condition
            # ------------------------------------------

            if active_cells == 0:

                print(

                    "\nTerrain Stable."

                )

                break

            if max_change < 1e-5:

                print(

                    "\nMaximum Change Below Threshold."

                )

                break

        return self.height
        # ==========================================================
    # Save Heightmap
    # ==========================================================

    def save_heightmap(

        self,

        filename,

        output_folder="OUTPUT"

    ):

        os.makedirs(

            output_folder,

            exist_ok=True

        )

        path = os.path.join(

            output_folder,

            filename

        )

        np.save(

            path,

            self.height

        )

        print(

            f"Saved : {path}"

        )


    # ==========================================================
    # Terrain Statistics
    # ==========================================================

    def statistics(self):

        print("\n" + "=" * 60)

        print("THERMAL EROSION REPORT")

        print("=" * 60)

        print(

            f"Iterations              : {self.thermal_iterations}"

        )

        print(

            f"Talus Angle             : {self.talus_angle:.2f}"

        )

        print(

            f"Thermal Strength        : {self.thermal_strength:.2f}"

        )

        print()

        print(

            f"Maximum Height Before   : {self.initial_max:.2f}"

        )

        print(

            f"Maximum Height After    : {np.max(self.height):.2f}"

        )

        print()

        print(

            f"Average Height Before   : {self.initial_mean:.2f}"

        )

        print(

            f"Average Height After    : {np.mean(self.height):.2f}"

        )

        print()

        print(

            f"Terrain Volume Before   : {self.initial_volume:.2f}"

        )

        print(

            f"Terrain Volume After    : {np.sum(self.height):.2f}"

        )

        print("=" * 60)


    # ==========================================================
    # Get Height Grid
    # ==========================================================

    def get_height(self):

        return self.height


    # ==========================================================
    # Run Thermal Erosion
    # ==========================================================

    def run(self):

        print("\nStarting Thermal Erosion...\n")

        self.thermal_erosion()

        self.statistics()

        self.save_heightmap(

            "thermal_height_grid.npy"

        )

        self.save_heightmap(

            "final_height_grid.npy"

        )

        print("\nThermal Erosion Completed.")

        return self.height
    