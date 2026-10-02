import os

import numpy as np


# ==========================================================
# HYDRAULIC EROSION ENGINE
# ==========================================================

class HydraulicErosion:

    # ======================================================
    # Constructor
    # ======================================================

    def __init__(

        self,

        height,

        hydraulic_iterations,

        rain_amount,

        evaporation,

        erosion_strength,

        deposition_strength,

        water_capacity

    ):

        # ----------------------------------------------
        # Terrain
        # ----------------------------------------------

        self.height = height.astype(np.float32)

        self.rows = self.height.shape[0]

        self.cols = self.height.shape[1]

        # ----------------------------------------------
        # Water Layer
        # ----------------------------------------------

        self.water = np.zeros(

            (

                self.rows,

                self.cols

            ),

            dtype=np.float32

        )

        # ----------------------------------------------
        # Sediment Layer
        # ----------------------------------------------

        self.sediment = np.zeros(

            (

                self.rows,

                self.cols

            ),

            dtype=np.float32

        )

        # ----------------------------------------------
        # Parameters
        # ----------------------------------------------

        self.hydraulic_iterations = hydraulic_iterations

        self.rain_amount = rain_amount

        self.evaporation = evaporation

        self.erosion_strength = erosion_strength

        self.deposition_strength = deposition_strength

        self.water_capacity = water_capacity

        # ----------------------------------------------
        # Output Folder
        # ----------------------------------------------

        self.output_folder = "OUTPUT"

        os.makedirs(

            self.output_folder,

            exist_ok=True

        )

        print("\nHydraulic Engine Initialized")

        print(

            "Grid Size :",

            self.rows,

            "x",

            self.cols

        )
            # ==========================================================
    # Rain Simulation
    # ==========================================================

    def add_rain(self):

        # ----------------------------------------------
        # Add uniform rainfall
        # ----------------------------------------------

        self.water += self.rain_amount

        # ----------------------------------------------
        # Statistics
        # ----------------------------------------------

        total_water = np.sum(self.water)

        average_water = np.mean(self.water)

        maximum_water = np.max(self.water)

        minimum_water = np.min(self.water)

        print(

            "\nRain Applied"

        )

        print(

            "Rain Amount :",

            self.rain_amount

        )

        print(

            "Total Water :",

            round(total_water, 3)

        )

        print(

            "Average Water :",

            round(average_water, 3)

        )

        print(

            "Maximum Water :",

            round(maximum_water, 3)

        )

        print(

            "Minimum Water :",

            round(minimum_water, 3)

        )
            # ==========================================================
    # Water Flow
    # ==========================================================

    def flow_water(self):

        water_delta = np.zeros_like(

            self.water,

            dtype=np.float32

        )

        # ------------------------------------------------------
        # Visit Every Cell
        # ------------------------------------------------------

        for r in range(1, self.rows - 1):

            for c in range(1, self.cols - 1):

                if self.water[r, c] <= 0:

                    continue

                current_level = (

                    self.height[r, c]

                    +

                    self.water[r, c]

                )

                neighbours = []

                # ----------------------------------------------
                # Find Lower Neighbours
                # ----------------------------------------------

                for nr in range(r - 1, r + 2):

                    for nc in range(c - 1, c + 2):

                        if nr == r and nc == c:

                            continue

                        neighbour_level = (

                            self.height[nr, nc]

                            +

                            self.water[nr, nc]

                        )

                        if neighbour_level < current_level:

                            neighbours.append(

                                (

                                    nr,

                                    nc,

                                    current_level - neighbour_level

                                )

                            )

                if len(neighbours) == 0:

                    continue

                # ----------------------------------------------
                # Weight by Height Difference
                # ----------------------------------------------

                total_difference = sum(

                    diff

                    for _, _, diff

                    in neighbours

                )

                available_water = self.water[r, c] * 0.50

                for nr, nc, diff in neighbours:

                    amount = (

                        available_water

                        *

                        (diff / total_difference)

                    )

                    water_delta[r, c] -= amount

                    water_delta[nr, nc] += amount

        # ------------------------------------------------------
        # Apply Flow
        # ------------------------------------------------------

        self.water += water_delta

        self.water = np.maximum(

            self.water,

            0.0

        )

        print(

            "Water Flow Completed"


        )
            # ==========================================================
    # Soil Erosion
    # ==========================================================

    def erode(self):

        eroded = 0.0

        for r in range(1, self.rows - 1):

            for c in range(1, self.cols - 1):

                # ------------------------------------------
                # Skip if no water
                # ------------------------------------------

                if self.water[r, c] <= 0:

                    continue

                # ------------------------------------------
                # Find Local Slope
                # ------------------------------------------

                current = self.height[r, c]

                max_drop = 0.0

                for nr in range(r - 1, r + 2):

                    for nc in range(c - 1, c + 2):

                        if nr == r and nc == c:

                            continue

                        drop = current - self.height[nr, nc]

                        if drop > max_drop:

                            max_drop = drop

                # ------------------------------------------
                # Skip Flat Terrain
                # ------------------------------------------

                if max_drop <= 0:

                    continue

                # ------------------------------------------
                # Sediment Capacity
                # ------------------------------------------

                capacity = (

                    self.water[r, c]

                    *

                    max_drop

                    *

                    self.water_capacity

                )

                # ------------------------------------------
                # Pick Up Soil
                # ------------------------------------------

                if self.sediment[r, c] < capacity:

                    amount = (

                        capacity

                        -

                        self.sediment[r, c]

                    )

                    amount *= self.erosion_strength

                    amount = min(

                        amount,

                        self.height[r, c] * 0.10

                    )

                    self.height[r, c] -= amount

                    self.sediment[r, c] += amount

                    eroded += amount

        print(

            "Soil Eroded :",

            round(

                eroded,

                3

            )

        )
            # ==========================================================
    # Sediment Transport
    # ==========================================================

    def transport(self):

        sediment_delta = np.zeros_like(

            self.sediment,

            dtype=np.float32

        )

        for r in range(1, self.rows - 1):

            for c in range(1, self.cols - 1):

                # ------------------------------------------
                # Skip if no sediment
                # ------------------------------------------

                if self.sediment[r, c] <= 0:

                    continue

                current_level = (

                    self.height[r, c]

                    +

                    self.water[r, c]

                )

                neighbours = []

                # ------------------------------------------
                # Find Lower Neighbours
                # ------------------------------------------

                for nr in range(r - 1, r + 2):

                    for nc in range(c - 1, c + 2):

                        if nr == r and nc == c:

                            continue

                        neighbour_level = (

                            self.height[nr, nc]

                            +

                            self.water[nr, nc]

                        )

                        if neighbour_level < current_level:

                            neighbours.append(

                                (

                                    nr,

                                    nc,

                                    current_level - neighbour_level

                                )

                            )

                if len(neighbours) == 0:

                    continue

                total_drop = sum(

                    drop

                    for _, _, drop

                    in neighbours

                )

                # ------------------------------------------
                # Move only 50% sediment
                # ------------------------------------------

                movable = self.sediment[r, c] * 0.50

                for nr, nc, drop in neighbours:

                    amount = (

                        movable

                        *

                        (drop / total_drop)

                    )

                    sediment_delta[r, c] -= amount

                    sediment_delta[nr, nc] += amount

        # ------------------------------------------
        # Apply Sediment Movement
        # ------------------------------------------

        self.sediment += sediment_delta

        self.sediment = np.maximum(

            self.sediment,

            0.0

        )

        print(

            "Sediment Transport Completed"

        )
            # ==========================================================
    # Sediment Deposition
    # ==========================================================

    def deposit(self):

        deposited = 0.0

        for r in range(1, self.rows - 1):

            for c in range(1, self.cols - 1):

                # ------------------------------------------
                # Skip dry cells
                # ------------------------------------------

                if self.water[r, c] <= 0:

                    continue

                # ------------------------------------------
                # Sediment Capacity
                # ------------------------------------------

                capacity = (

                    self.water[r, c]

                    *

                    self.water_capacity

                )

                # ------------------------------------------
                # Deposit if carrying too much sediment
                # ------------------------------------------

                if self.sediment[r, c] > capacity:

                    excess = (

                        self.sediment[r, c]

                        -

                        capacity

                    )

                    amount = (

                        excess

                        *

                        self.deposition_strength

                    )

                    self.height[r, c] += amount

                    self.sediment[r, c] -= amount

                    deposited += amount

        print(

            "Sediment Deposited :",

            round(

                deposited,

                3

            )

        )
            # ==========================================================
    # Evaporation
    # ==========================================================

    def evaporate(self):

        self.water *= (

            1.0 - self.evaporation

        )

        self.water = np.maximum(

            self.water,

            0.0

        )

        print(

            "Evaporation Applied"

        )


    # ==========================================================
    # Statistics
    # ==========================================================

    def statistics(self):

        print("\n" + "=" * 60)

        print("HYDRAULIC STATISTICS")

        print("=" * 60)

        print(

            "Maximum Height :",

            np.max(self.height)

        )

        print(

            "Minimum Height :",

            np.min(self.height)

        )

        print(

            "Average Height :",

            round(

                np.mean(self.height),

                3

            )

        )

        print(

            "Total Water :",

            round(

                np.sum(self.water),

                3

            )

        )

        print(

            "Total Sediment :",

            round(

                np.sum(self.sediment),

                3

            )

        )


    # ==========================================================
    # Save Heightmap
    # ==========================================================

    def save_heightmap(

        self,

        filename

    ):

        path = os.path.join(

            self.output_folder,

            filename

        )

        np.save(

            path,

            self.height

        )

        print(

            "Saved :", filename

        )


    # ==========================================================
    # Run Simulation
    # ==========================================================

    def run(self):

        print("\n" + "=" * 60)

        print("HYDRAULIC EROSION")

        print("=" * 60)

        for i in range(

            self.hydraulic_iterations

        ):

            print(

                f"\nIteration {i+1}/{self.hydraulic_iterations}"

            )

            self.add_rain()

            self.flow_water()

            self.erode()

            self.transport()

            self.deposit()

            self.evaporate()

        self.statistics()

        self.save_heightmap(

            "hydraulic_height_grid.npy"

        )

        self.save_heightmap(

            "final_height_grid.npy"

        )

        return self.height