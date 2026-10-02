import os
import numpy as np

from ca_state import (
    EMPTY,
    WATER,
    SEDIMENT,
    SOIL,
    SLOPE,
    ROCK,
    PLATEAU,
    PEAK,
    STATE_NAMES
)


# ==========================================================
# PART 6
# R4 - GRID BASED HYDRAULIC EROSION
# ==========================================================


class CAHydraulicErosion:

    def __init__(
        self,
        height_grid,
        state_grid,
        sediment_grid,
        iterations=100,
        rain_amount=0.20,
        rain_frequency=1,
        evaporation_rate=0.02,
        flow_rate=0.60,
        carrying_capacity=0.80,
        erosion_rate=0.15,
        deposition_rate=0.25,
        soil_softness=0.30,
        max_erosion=0.30,
        max_deposition=0.30
    ):

        # --------------------------------------------------
        # Terrain
        # --------------------------------------------------

        self.height = (
            height_grid
            .astype(np.float32)
            .copy()
        )

        # --------------------------------------------------
        # CA states
        # --------------------------------------------------

        self.state = (
            state_grid
            .astype(np.uint8)
            .copy()
        )

        # --------------------------------------------------
        # Existing sediment from R3
        # --------------------------------------------------

        self.sediment = (
            sediment_grid
            .astype(np.float32)
            .copy()
        )

        # --------------------------------------------------
        # Water field
        # --------------------------------------------------

        self.water = np.zeros(
            self.height.shape,
            dtype=np.float32
        )

        # --------------------------------------------------
        # Original terrain
        # --------------------------------------------------

        self.original_height = (
            self.height.copy()
        )

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------

        self.iterations = iterations

        self.rain_amount = rain_amount

        self.rain_frequency = rain_frequency

        self.evaporation_rate = (
            evaporation_rate
        )

        self.flow_rate = flow_rate

        self.carrying_capacity = (
            carrying_capacity
        )

        self.erosion_rate = erosion_rate

        self.deposition_rate = (
            deposition_rate
        )

        self.soil_softness = (
            soil_softness
        )

        self.max_erosion = max_erosion

        self.max_deposition = (
            max_deposition
        )

        self.rows, self.cols = (
            self.height.shape
        )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        self.total_rain = 0.0

        self.total_water_moved = 0.0

        self.total_hydraulic_erosion = 0.0

        self.total_hydraulic_deposition = 0.0

        self.erosion_events = 0

        self.deposition_events = 0

        self.state_changes = 0

    # ======================================================
    # Moore Neighbourhood
    # ======================================================

    def neighbours(self, r, c):

        result = []

        for dr in (-1, 0, 1):

            for dc in (-1, 0, 1):

                if dr == 0 and dc == 0:
                    continue

                nr = r + dr
                nc = c + dc

                if nr < 0:
                    continue

                if nr >= self.rows:
                    continue

                if nc < 0:
                    continue

                if nc >= self.cols:
                    continue

                result.append((nr, nc))

        return result

    # ======================================================
    # State erosion resistance
    # ======================================================

    def erosion_resistance(self, state):

        if state == SOIL:
            return 1.00

        if state == SLOPE:
            return 0.75

        if state == SEDIMENT:
            return 1.20

        if state == ROCK:
            return 0.20

        if state == PLATEAU:
            return 0.10

        if state == PEAK:
            return 0.05

        return 0.0

    # ======================================================
    # Rainfall
    # ======================================================

    def apply_rainfall(self):

        # --------------------------------------------------
        # Add water to terrain cells
        # --------------------------------------------------

        terrain_mask = (
            self.height > 0
        )

        rain = np.zeros_like(
            self.water
        )

        rain[terrain_mask] = (
            self.rain_amount
        )

        self.water += rain

        self.total_rain += np.sum(
            rain
        )

    # ======================================================
    # Calculate hydraulic flow
    # ======================================================

    def calculate_flow(self):

        water_delta = np.zeros(
            self.water.shape,
            dtype=np.float32
        )

        total_moved = 0.0

        # --------------------------------------------------
        # Water surface = terrain + water
        # --------------------------------------------------

        water_surface = (
            self.height
            +
            self.water
        )

        # --------------------------------------------------
        # Process every interior cell
        # --------------------------------------------------

        for r in range(
            1,
            self.rows - 1
        ):

            for c in range(
                1,
                self.cols - 1
            ):

                current_water = (
                    self.water[r, c]
                )

                if current_water <= 0:

                    continue

                current_surface = (
                    water_surface[r, c]
                )

                lower = []

                # --------------------------------------------------
                # Find lower water-surface neighbours
                # --------------------------------------------------

                for nr, nc in self.neighbours(
                    r,
                    c
                ):

                    neighbour_surface = (
                        water_surface[nr, nc]
                    )

                    difference = (
                        current_surface
                        -
                        neighbour_surface
                    )

                    if difference > 0:

                        lower.append(
                            (
                                nr,
                                nc,
                                difference
                            )
                        )

                if len(lower) == 0:

                    continue

                # --------------------------------------------------
                # Total gradient
                # --------------------------------------------------

                total_difference = sum(
                    d
                    for _, _, d in lower
                )

                if total_difference <= 0:

                    continue

                # --------------------------------------------------
                # Maximum amount available
                # --------------------------------------------------

                movable_water = (
                    current_water
                    *
                    self.flow_rate
                )

                movable_water = min(
                    movable_water,
                    current_water
                )

                # --------------------------------------------------
                # Distribute water proportionally
                # --------------------------------------------------

                for nr, nc, difference in lower:

                    fraction = (
                        difference
                        /
                        total_difference
                    )

                    amount = (
                        movable_water
                        *
                        fraction
                    )

                    water_delta[r, c] -= (
                        amount
                    )

                    water_delta[nr, nc] += (
                        amount
                    )

                    total_moved += amount

        # --------------------------------------------------
        # Apply simultaneously
        # --------------------------------------------------

        self.water += water_delta

        self.total_water_moved += (
            total_moved
        )

        return total_moved

    # ======================================================
    # Hydraulic erosion + deposition
    # ======================================================

    def hydraulic_material_process(self):

        height_delta = np.zeros(
            self.height.shape,
            dtype=np.float32
        )

        sediment_delta = np.zeros(
            self.sediment.shape,
            dtype=np.float32
        )

        iteration_erosion = 0.0

        iteration_deposition = 0.0

        iteration_erosion_events = 0

        iteration_deposition_events = 0

        # --------------------------------------------------
        # Calculate water gradients
        # --------------------------------------------------

        for r in range(
            1,
            self.rows - 1
        ):

            for c in range(
                1,
                self.cols - 1
            ):

                water_amount = (
                    self.water[r, c]
                )

                if water_amount <= 0:

                    continue

                current_height = (
                    self.height[r, c]
                )

                current_state = int(
                    self.state[r, c]
                )

                # --------------------------------------------------
                # Find steepest downhill terrain direction
                # --------------------------------------------------

                lower = []

                for nr, nc in self.neighbours(
                    r,
                    c
                ):

                    drop = (
                        current_height
                        -
                        self.height[nr, nc]
                    )

                    if drop > 0:

                        lower.append(
                            (
                                nr,
                                nc,
                                drop
                            )
                        )

                if len(lower) == 0:

                    # --------------------------------------------------
                    # Water has nowhere to flow.
                    # This is a deposition location.
                    # --------------------------------------------------

                    capacity = 0.0

                else:

                    steepest = max(
                        lower,
                        key=lambda x: x[2]
                    )

                    steepest_drop = (
                        steepest[2]
                    )

                    # --------------------------------------------------
                    # Hydraulic carrying capacity
                    #
                    # capacity increases with:
                    # water amount
                    # slope/drop
                    # carrying capacity constant
                    # --------------------------------------------------

                    capacity = (
                        self.carrying_capacity
                        *
                        water_amount
                        *
                        steepest_drop
                    )

                current_sediment = (
                    self.sediment[r, c]
                )

                # ==================================================
                # CASE 1
                # Too much sediment -> deposition
                # ==================================================

                if current_sediment > capacity:

                    excess = (
                        current_sediment
                        -
                        capacity
                    )

                    deposit = (
                        excess
                        *
                        self.deposition_rate
                    )

                    deposit = min(
                        deposit,
                        self.max_deposition
                    )

                    if deposit > 0:

                        sediment_delta[r, c] -= (
                            deposit
                        )

                        height_delta[r, c] += (
                            deposit
                        )

                        iteration_deposition += (
                            deposit
                        )

                        iteration_deposition_events += 1

                        # --------------------------------------------------
                        # Sediment deposition produces soil
                        # --------------------------------------------------

                        if current_state in (
                            WATER,
                            SEDIMENT
                        ):

                            self.state[r, c] = (
                                SOIL
                            )

                            self.state_changes += 1

                # ==================================================
                # CASE 2
                # Water can carry more sediment -> erosion
                # ==================================================

                elif current_sediment < capacity:

                    capacity_gap = (
                        capacity
                        -
                        current_sediment
                    )

                    resistance = (
                        self.erosion_resistance(
                            current_state
                        )
                    )

                    if resistance <= 0:

                        continue

                    erosion_amount = (
                        capacity_gap
                        *
                        self.erosion_rate
                        *
                        resistance
                    )

                    erosion_amount = min(
                        erosion_amount,
                        self.max_erosion
                    )

                    # --------------------------------------------------
                    # Never remove too much
                    # --------------------------------------------------

                    erosion_amount = min(
                        erosion_amount,
                        max(
                            0.0,
                            current_height
                            -
                            0.10
                        )
                    )

                    if erosion_amount <= 0:

                        continue

                    # --------------------------------------------------
                    # Remove terrain
                    # --------------------------------------------------

                    height_delta[r, c] -= (
                        erosion_amount
                    )

                    # --------------------------------------------------
                    # Add to suspended sediment
                    # --------------------------------------------------

                    sediment_delta[r, c] += (
                        erosion_amount
                    )

                    iteration_erosion += (
                        erosion_amount
                    )

                    iteration_erosion_events += 1

                    # --------------------------------------------------
                    # Strongly eroded SOIL/SLOPE becomes SEDIMENT
                    # --------------------------------------------------

                    if current_state in (
                        SOIL,
                        SLOPE
                    ):

                        if erosion_amount > 0.05:

                            self.state[r, c] = (
                                SEDIMENT
                            )

                            self.state_changes += 1

        # --------------------------------------------------
        # Apply simultaneously
        # --------------------------------------------------

        self.height += height_delta

        self.sediment += sediment_delta

        self.total_hydraulic_erosion += (
            iteration_erosion
        )

        self.total_hydraulic_deposition += (
            iteration_deposition
        )

        self.erosion_events += (
            iteration_erosion_events
        )

        self.deposition_events += (
            iteration_deposition_events
        )

        return (
            iteration_erosion,
            iteration_deposition,
            iteration_erosion_events,
            iteration_deposition_events
        )

    # ======================================================
    # Evaporation
    # ======================================================

    def evaporate(self):

        before = np.sum(
            self.water
        )

        self.water *= (
            1.0
            -
            self.evaporation_rate
        )

        after = np.sum(
            self.water
        )

        return before - after

    # ======================================================
    # One complete hydraulic iteration
    # ======================================================

    def step(self):

        # --------------------------------------------------
        # 1. Rain
        # --------------------------------------------------

        if (
            self.current_iteration
            %
            self.rain_frequency
            == 0
        ):

            self.apply_rainfall()

        # --------------------------------------------------
        # 2. Hydraulic material interaction
        #
        # We process material before the next flow pass
        # so that the terrain can evolve continuously.
        # --------------------------------------------------

        (
            erosion,
            deposition,
            erosion_events,
            deposition_events
        ) = self.hydraulic_material_process()

        # --------------------------------------------------
        # 3. Water flow
        # --------------------------------------------------

        moved_water = (
            self.calculate_flow()
        )

        # --------------------------------------------------
        # 4. Evaporation
        # --------------------------------------------------

        evaporated = (
            self.evaporate()
        )

        return (
            erosion,
            deposition,
            moved_water,
            evaporated,
            erosion_events,
            deposition_events
        )

    # ======================================================
    # Run simulation
    # ======================================================

    def run(self):

        print("\n" + "=" * 60)

        print(
            "R4 - GRID BASED HYDRAULIC EROSION"
        )

        print("=" * 60)

        print(
            "\nIterations :",
            self.iterations
        )

        print(
            "Rain Amount :",
            self.rain_amount
        )

        print(
            "Rain Frequency :",
            self.rain_frequency
        )

        print(
            "Evaporation Rate :",
            self.evaporation_rate
        )

        print(
            "Flow Rate :",
            self.flow_rate
        )

        print(
            "Carrying Capacity :",
            self.carrying_capacity
        )

        print(
            "Erosion Rate :",
            self.erosion_rate
        )

        print(
            "Deposition Rate :",
            self.deposition_rate
        )

        self.current_iteration = 0

        for i in range(
            self.iterations
        ):

            self.current_iteration = (
                i + 1
            )

            (
                erosion,
                deposition,
                moved_water,
                evaporated,
                erosion_events,
                deposition_events
            ) = self.step()

            print(
                f"Iteration "
                f"{i + 1:3d} | "
                f"Rain: "
                f"{self.rain_amount:.3f} | "
                f"Water Moved: "
                f"{moved_water:.4f} | "
                f"Hydraulic Erosion: "
                f"{erosion:.4f} | "
                f"Deposition: "
                f"{deposition:.4f} | "
                f"Evaporated: "
                f"{evaporated:.4f}"
            )

        return (
            self.height,
            self.state,
            self.water,
            self.sediment
        )

    # ======================================================
    # Statistics
    # ======================================================

    def statistics(self):

        print("\n" + "=" * 60)

        print(
            "R4 HYDRAULIC STATISTICS"
        )

        print("=" * 60)

        height_change = (
            self.height
            -
            self.original_height
        )

        print(
            "\nTotal Rain :",
            self.total_rain
        )

        print(
            "Total Water Moved :",
            self.total_water_moved
        )

        print(
            "Total Hydraulic Erosion :",
            self.total_hydraulic_erosion
        )

        print(
            "Total Hydraulic Deposition :",
            self.total_hydraulic_deposition
        )

        print(
            "Erosion Events :",
            self.erosion_events
        )

        print(
            "Deposition Events :",
            self.deposition_events
        )

        print(
            "State Changes :",
            self.state_changes
        )

        print(
            "\nFinal Water Volume :",
            np.sum(self.water)
        )

        print(
            "Final Sediment :",
            np.sum(self.sediment)
        )

        print(
            "Maximum Height Loss :",
            abs(
                np.min(
                    height_change
                )
            )
        )

        print(
            "Maximum Height Gain :",
            np.max(
                height_change
            )
        )

        print(
            "Mean Absolute Height Change :",
            np.mean(
                np.abs(
                    height_change
                )
            )
        )

        print(
            "\nFinal Height Range :",
            np.min(self.height),
            "to",
            np.max(self.height)
        )

        print(
            "\nFinal State Counts:"
        )

        for state in range(8):

            count = np.sum(
                self.state == state
            )

            print(
                f"{state} - "
                f"{STATE_NAMES[state]:10s} : "
                f"{count}"
            )

    # ======================================================
    # Save
    # ======================================================

    def save(self):

        os.makedirs(
            "OUTPUT",
            exist_ok=True
        )

        np.save(
            "OUTPUT/hydraulic_height_grid.npy",
            self.height
        )

        np.save(
            "OUTPUT/hydraulic_state_grid.npy",
            self.state
        )

        np.save(
            "OUTPUT/water_grid.npy",
            self.water
        )

        np.save(
            "OUTPUT/hydraulic_sediment_grid.npy",
            self.sediment
        )

        print(
            "\nGenerated Files:"
        )

        print(
            "OUTPUT/hydraulic_height_grid.npy"
        )

        print(
            "OUTPUT/hydraulic_state_grid.npy"
        )

        print(
            "OUTPUT/water_grid.npy"
        )

        print(
            "OUTPUT/hydraulic_sediment_grid.npy"
        )


# ==========================================================
# STANDALONE EXECUTION
# ==========================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        "CA TERRAIN - R4 HYDRAULIC EROSION"
    )

    print("=" * 60)

    # ------------------------------------------------------
    # R3 outputs
    # ------------------------------------------------------

    height_path = os.path.join(
        "OUTPUT",
        "transported_height_grid.npy"
    )

    state_path = os.path.join(
        "OUTPUT",
        "transported_state_grid.npy"
    )

    sediment_path = os.path.join(
        "OUTPUT",
        "final_sediment_grid.npy"
    )

    # ------------------------------------------------------
    # Check files
    # ------------------------------------------------------

    if not os.path.exists(
        height_path
    ):

        raise FileNotFoundError(
            "Run ca_sediment.py first."
        )

    if not os.path.exists(
        state_path
    ):

        raise FileNotFoundError(
            "Run ca_sediment.py first."
        )

    if not os.path.exists(
        sediment_path
    ):

        raise FileNotFoundError(
            "Run ca_sediment.py first."
        )

    # ------------------------------------------------------
    # Load R3 output
    # ------------------------------------------------------

    height_grid = np.load(
        height_path
    )

    state_grid = np.load(
        state_path
    )

    sediment_grid = np.load(
        sediment_path
    )

    print(
        "\nInput Grid :",
        height_grid.shape
    )

    print(
        "Input Maximum Height :",
        np.max(height_grid)
    )

    print(
        "Input Minimum Height :",
        np.min(height_grid)
    )

    print(
        "Initial Sediment :",
        np.sum(sediment_grid)
    )

    # ------------------------------------------------------
    # Parameters
    # ------------------------------------------------------

    iterations = int(
        input(
            "\nHydraulic Iterations : "
        )
    )

    if iterations <= 0:

        raise ValueError(
            "Iterations must be greater than zero."
        )

    rain_amount = float(
        input(
            "Rain Amount : "
        )
    )

    if rain_amount <= 0:

        raise ValueError(
            "Rain Amount must be greater than zero."
        )

    rain_frequency = int(
        input(
            "Rain Frequency : "
        )
    )

    if rain_frequency <= 0:

        raise ValueError(
            "Rain Frequency must be greater than zero."
        )

    evaporation_rate = float(
        input(
            "Evaporation Rate (0-1) : "
        )
    )

    if not (
        0 <= evaporation_rate < 1
    ):

        raise ValueError(
            "Evaporation Rate must be between 0 and 1."
        )

    flow_rate = float(
        input(
            "Flow Rate (0-1) : "
        )
    )

    if not (
        0 < flow_rate <= 1
    ):

        raise ValueError(
            "Flow Rate must be between 0 and 1."
        )

    carrying_capacity = float(
        input(
            "Carrying Capacity : "
        )
    )

    if carrying_capacity <= 0:

        raise ValueError(
            "Carrying Capacity must be greater than zero."
        )

    erosion_rate = float(
        input(
            "Hydraulic Erosion Rate (0-1) : "
        )
    )

    if not (
        0 < erosion_rate <= 1
    ):

        raise ValueError(
            "Erosion Rate must be between 0 and 1."
        )

    deposition_rate = float(
        input(
            "Hydraulic Deposition Rate (0-1) : "
        )
    )

    if not (
        0 < deposition_rate <= 1
    ):

        raise ValueError(
            "Deposition Rate must be between 0 and 1."
        )

    soil_softness = float(
        input(
            "Soil Softness (0-1) : "
        )
    )

    if not (
        0 < soil_softness <= 1
    ):

        raise ValueError(
            "Soil Softness must be between 0 and 1."
        )

    max_erosion = float(
        input(
            "Maximum Hydraulic Erosion : "
        )
    )

    if max_erosion <= 0:

        raise ValueError(
            "Maximum Hydraulic Erosion must be greater than zero."
        )

    max_deposition = float(
        input(
            "Maximum Hydraulic Deposition : "
        )
    )

    if max_deposition <= 0:

        raise ValueError(
            "Maximum Hydraulic Deposition must be greater than zero."
        )

    # ------------------------------------------------------
    # Create hydraulic engine
    # ------------------------------------------------------

    hydraulic = CAHydraulicErosion(

        height_grid,

        state_grid,

        sediment_grid,

        iterations=iterations,

        rain_amount=rain_amount,

        rain_frequency=rain_frequency,

        evaporation_rate=evaporation_rate,

        flow_rate=flow_rate,

        carrying_capacity=carrying_capacity,

        erosion_rate=erosion_rate,

        deposition_rate=deposition_rate,

        soil_softness=soil_softness,

        max_erosion=max_erosion,

        max_deposition=max_deposition
    )

    # ------------------------------------------------------
    # Run
    # ------------------------------------------------------

    hydraulic.run()

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    hydraulic.statistics()

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    hydraulic.save()

    print(
        "\nR4 Hydraulic Erosion Completed."
    )