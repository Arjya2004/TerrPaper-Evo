import os
import numpy as np


# ==========================================================
# R5.4
# STATE-AWARE TERRAIN MATERIAL COUPLING
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
# FINAL TERRAIN CA
# ==========================================================

class FinalTerrainCA:

    def __init__(
        self,
        height_grid,
        state_grid,
        sediment_grid,
        iterations=30,
        slope_threshold=0.45,
        steep_threshold=0.70,
        plateau_threshold=0.20,
        peak_threshold=0.80,
        sediment_threshold=0.05
    ):

        self.height = (
            height_grid.astype(np.float32).copy()
        )

        self.state = (
            state_grid.astype(np.uint8).copy()
        )

        self.sediment = (
            sediment_grid.astype(np.float32).copy()
        )

        self.original_height = (
            self.height.copy()
        )

        self.original_state = (
            self.state.copy()
        )

        self.original_sediment = (
            self.sediment.copy()
        )

        self.iterations = iterations

        self.slope_threshold = slope_threshold
        self.steep_threshold = steep_threshold
        self.plateau_threshold = plateau_threshold
        self.peak_threshold = peak_threshold
        self.sediment_threshold = sediment_threshold

        # ==================================================
        # R5.4 MATERIAL PARAMETERS
        # ==================================================

        # Soil erosion
        self.soil_erosion_rate = 0.025

        # Slope erosion
        self.slope_erosion_rate = 0.040

        # Maximum erosion per cell
        self.max_erosion = 0.12

        # Fraction of sediment transported downhill
        self.transport_rate = 0.18

        # Deposition rate
        self.deposition_rate = 0.25

        # Minimum sediment required to activate
        # SEDIMENT state
        self.sediment_activation = 0.025

        # Amount below which sediment becomes stable soil
        self.sediment_clear_threshold = 0.008

        # Height safety
        self.minimum_height = 0.01

        self.rows, self.cols = (
            self.height.shape
        )

        self.total_state_changes = 0
        self.total_erosion = 0.0
        self.total_deposition = 0.0
        self.total_transport = 0.0

        self.transition_counts = {}


    # ======================================================
    # MOORE NEIGHBOURS
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

                result.append(
                    (nr, nc)
                )

        return result


    # ======================================================
    # SLOPE
    # ======================================================

    def calculate_slope(self):

        gy, gx = np.gradient(
            self.height
        )

        return np.sqrt(
            gx * gx +
            gy * gy
        )


    # ======================================================
    # NORMALIZED HEIGHT
    # ======================================================

    def normalize_height(self):

        terrain = self.height[
            self.height > 0
        ]

        if len(terrain) == 0:

            return np.zeros_like(
                self.height
            )

        h_min = np.min(
            terrain
        )

        h_max = np.max(
            terrain
        )

        if h_max == h_min:

            return np.zeros_like(
                self.height
            )

        return (
            self.height - h_min
        ) / (
            h_max - h_min
        )


    # ======================================================
    # NORMALIZED SLOPE
    # ======================================================

    def normalize_slope(
        self,
        slope
    ):

        terrain = slope[
            self.height > 0
        ]

        if len(terrain) == 0:

            return np.zeros_like(
                slope
            )

        maximum = np.max(
            terrain
        )

        if maximum <= 0:

            return np.zeros_like(
                slope
            )

        return (
            slope / maximum
        )


    # ======================================================
    # FIND LOWEST NEIGHBOUR
    # ======================================================

    def lowest_neighbour(
        self,
        r,
        c
    ):

        current_height = float(
            self.height[r, c]
        )

        best = None
        best_height = current_height

        for nr, nc in self.neighbours(
            r,
            c
        ):

            neighbour_height = float(
                self.height[nr, nc]
            )

            if (
                neighbour_height
                <
                best_height
            ):

                best_height = (
                    neighbour_height
                )

                best = (
                    nr,
                    nc
                )

        return best


    # ======================================================
    # NEIGHBOUR SEDIMENT COUNT
    # ======================================================

    def sediment_neighbour_count(
        self,
        r,
        c
    ):

        count = 0

        for nr, nc in self.neighbours(
            r,
            c
        ):

            if (
                self.state[nr, nc]
                ==
                SEDIMENT
            ):

                count += 1

        return count


    # ======================================================
    # NEIGHBOUR SEDIMENT AMOUNT
    # ======================================================

    def sediment_neighbour_amount(
        self,
        r,
        c
    ):

        total = 0.0

        for nr, nc in self.neighbours(
            r,
            c
        ):

            total += float(
                self.sediment[nr, nc]
            )

        return total


    # ======================================================
    # RECORD TRANSITION
    # ======================================================

    def record_transition(
        self,
        old_state,
        new_state
    ):

        if old_state == new_state:
            return

        key = (
            STATE_NAMES[old_state],
            STATE_NAMES[new_state]
        )

        self.transition_counts[key] = (
            self.transition_counts.get(
                key,
                0
            ) + 1
        )


    # ======================================================
    # MATERIAL PASS
    # ======================================================

    def material_pass(self):

        slope = (
            self.calculate_slope()
        )

        normalized_slope = (
            self.normalize_slope(
                slope
            )
        )

        new_height = (
            self.height.copy()
        )

        new_sediment = (
            self.sediment.copy()
        )

        erosion_grid = (
            np.zeros_like(
                self.height,
                dtype=np.float32
            )
        )

        deposition_grid = (
            np.zeros_like(
                self.height,
                dtype=np.float32
            )
        )

        transport_grid = (
            np.zeros_like(
                self.sediment,
                dtype=np.float32
            )
        )


        # ==================================================
        # PASS A
        # EROSION
        # ==================================================

        for r in range(
            1,
            self.rows - 1
        ):

            for c in range(
                1,
                self.cols - 1
            ):

                state = int(
                    self.state[r, c]
                )

                local_slope = float(
                    normalized_slope[r, c]
                )

                erosion = 0.0


                # ------------------------------------------
                # SOIL
                # ------------------------------------------

                if state == SOIL:

                    if (
                        local_slope
                        >
                        self.slope_threshold
                    ):

                        excess = (
                            local_slope
                            -
                            self.slope_threshold
                        )

                        erosion = (
                            excess
                            *
                            self.soil_erosion_rate
                        )


                # ------------------------------------------
                # SLOPE
                # ------------------------------------------

                elif state == SLOPE:

                    if (
                        local_slope
                        >
                        self.slope_threshold
                    ):

                        excess = (
                            local_slope
                            -
                            self.slope_threshold
                        )

                        erosion = (
                            excess
                            *
                            self.slope_erosion_rate
                        )


                # ------------------------------------------
                # LIMIT
                # ------------------------------------------

                erosion = min(
                    erosion,
                    self.max_erosion
                )

                erosion = min(
                    erosion,
                    max(
                        0.0,
                        float(
                            self.height[r, c]
                        )
                        -
                        self.minimum_height
                    )
                )


                if erosion > 0:

                    erosion_grid[r, c] = (
                        erosion
                    )

                    new_height[r, c] -= (
                        erosion
                    )

                    new_sediment[r, c] += (
                        erosion
                    )


        # ==================================================
        # PASS B
        # SEDIMENT TRANSPORT
        # ==================================================

        for r in range(
            1,
            self.rows - 1
        ):

            for c in range(
                1,
                self.cols - 1
            ):

                sediment = float(
                    new_sediment[r, c]
                )

                if sediment <= 0:
                    continue

                state = int(
                    self.state[r, c]
                )

                # Only actively eroding terrain
                # transports material.
                if state not in (
                    SLOPE,
                    SEDIMENT
                ):
                    continue

                target = (
                    self.lowest_neighbour(
                        r,
                        c
                    )
                )

                if target is None:
                    continue

                nr, nc = target

                height_difference = (
                    float(
                        self.height[r, c]
                    )
                    -
                    float(
                        self.height[nr, nc]
                    )
                )

                if height_difference <= 0:
                    continue

                transport = min(
                    sediment
                    *
                    self.transport_rate,
                    0.05
                    +
                    height_difference
                    *
                    0.02
                )

                transport = min(
                    transport,
                    sediment
                )

                if transport <= 0:
                    continue

                new_sediment[r, c] -= (
                    transport
                )

                transport_grid[nr, nc] += (
                    transport
                )

                self.total_transport += (
                    transport
                )


        # Apply transported sediment
        new_sediment += (
            transport_grid
        )


        # ==================================================
        # PASS C
        # DEPOSITION
        # ==================================================

        for r in range(
            1,
            self.rows - 1
        ):

            for c in range(
                1,
                self.cols - 1
            ):

                sediment = float(
                    new_sediment[r, c]
                )

                if sediment <= 0:
                    continue

                local_slope = float(
                    normalized_slope[r, c]
                )

                # Deposition mainly occurs
                # in flatter regions.
                if (
                    local_slope
                    <
                    self.slope_threshold
                ):

                    flatness = (
                        self.slope_threshold
                        -
                        local_slope
                    ) / self.slope_threshold

                    deposition = (
                        sediment
                        *
                        self.deposition_rate
                        *
                        flatness
                    )

                    deposition = min(
                        deposition,
                        sediment
                    )

                    if deposition > 0:

                        new_height[r, c] += (
                            deposition
                        )

                        new_sediment[r, c] -= (
                            deposition
                        )

                        deposition_grid[r, c] = (
                            deposition
                        )


        # ==================================================
        # VALIDATE
        # ==================================================

        new_height = np.maximum(
            new_height,
            self.minimum_height
        )

        new_sediment = np.maximum(
            new_sediment,
            0.0
        )


        # ==================================================
        # APPLY
        # ==================================================

        self.height = (
            new_height
        )

        self.sediment = (
            new_sediment
        )


        return (
            normalized_slope,
            erosion_grid,
            deposition_grid
        )


    # ======================================================
    # R5.5 FINAL STATE TRANSITION
    # ======================================================

    def state_transition(
        self,
        normalized_slope,
        erosion_grid,
        deposition_grid
    ):

        normalized_height = (
            self.normalize_height()
        )

        new_state = self.state.copy()

        state_changes = 0

        for r in range(1, self.rows - 1):

            for c in range(1, self.cols - 1):

                current = int(
                    self.state[r, c]
                )

                slope = float(
                    normalized_slope[r, c]
                )

                height = float(
                    normalized_height[r, c]
                )

                sediment = float(
                    self.sediment[r, c]
                )

                erosion = float(
                    erosion_grid[r, c]
                )

                deposition = float(
                    deposition_grid[r, c]
                )

                sediment_neighbours = (
                    self.sediment_neighbour_count(
                        r,
                        c
                    )
                )

                new = current


                # ==================================================
                # SOIL -> SLOPE
                # ==================================================

                if current == SOIL:

                    if (
                        slope >=
                        self.slope_threshold
                    ):

                        new = SLOPE

                    elif (
                        height >= 0.55
                        and
                        slope <
                        self.plateau_threshold
                    ):

                        new = PLATEAU


                # ==================================================
                # SLOPE
                # ==================================================

                elif current == SLOPE:

                    # ----------------------------------------------
                    # R5.5 IMPORTANT RULE
                    #
                    # Slope becomes sediment ONLY when:
                    #
                    # 1. actual erosion is happening
                    # AND
                    # 2. enough sediment exists
                    #
                    # OR strong sediment neighbourhood exists.
                    # ----------------------------------------------

                    active_erosion = (
                        erosion >= 0.005
                    )

                    active_sediment = (
                        sediment >=
                        self.sediment_activation
                    )

                    strong_sediment_zone = (
                        sediment_neighbours >= 2
                        and
                        sediment >= 0.015
                    )


                    if (
                        active_erosion
                        and
                        active_sediment
                    ):

                        new = SEDIMENT

                    elif strong_sediment_zone:

                        new = SEDIMENT

                    elif (
                        height >=
                        self.peak_threshold
                        and
                        slope >=
                        self.steep_threshold
                    ):

                        new = ROCK

                    elif (
                        height >= 0.55
                        and
                        slope <
                        self.plateau_threshold
                    ):

                        new = PLATEAU

                    elif (
                        slope <
                        0.35
                        and
                        sediment <
                        self.sediment_clear_threshold
                    ):

                        new = SOIL


                # ==================================================
                # SEDIMENT -> SOIL
                # ==================================================

                elif current == SEDIMENT:

                    # ----------------------------------------------
                    # Stable deposition
                    # ----------------------------------------------

                    if (
                        sediment <=
                        self.sediment_clear_threshold
                        and
                        slope <
                        self.slope_threshold
                    ):

                        new = SOIL


                    # ----------------------------------------------
                    # Deposition is occurring and sediment is
                    # being consumed -> SOIL can form.
                    # ----------------------------------------------

                    elif (
                        deposition > 0
                        and
                        sediment <
                        self.sediment_activation
                    ):

                        new = SOIL


                    # ----------------------------------------------
                    # Active sediment remains sediment.
                    # ----------------------------------------------

                    elif (
                        sediment >=
                        self.sediment_activation
                    ):

                        new = SEDIMENT


                    # ----------------------------------------------
                    # Weak sediment on gentle terrain
                    # eventually becomes soil.
                    # ----------------------------------------------

                    elif (
                        slope <
                        0.35
                    ):

                        new = SOIL


                # ==================================================
                # ROCK
                # ==================================================

                elif current == ROCK:

                    if (
                        height >=
                        self.peak_threshold
                    ):

                        new = PEAK

                    elif (
                        slope <
                        self.plateau_threshold
                        and
                        height >= 0.55
                    ):

                        new = PLATEAU


                # ==================================================
                # PLATEAU
                # ==================================================

                elif current == PLATEAU:

                    if (
                        height >=
                        self.peak_threshold
                    ):

                        new = PEAK

                    elif (
                        slope >=
                        self.steep_threshold
                    ):

                        new = ROCK


                # ==================================================
                # PEAK
                # ==================================================

                elif current == PEAK:

                    if (
                        height <
                        0.70
                    ):

                        new = PLATEAU


                # ==================================================
                # APPLY TRANSITION
                # ==================================================

                if new != current:

                    new_state[r, c] = new

                    self.record_transition(
                        current,
                        new
                    )

                    state_changes += 1


        # ==================================================
        # SYNCHRONOUS STATE UPDATE
        # ==================================================

        self.state = new_state

        self.total_state_changes += (
            state_changes
        )

        return state_changes

   # ======================================================
    # RUN
    # ======================================================

    def run(self):

        print(
            "\n" + "=" * 65
        )

        print(
            "R5.5 - FINAL STATE AWARE MATERIAL COUPLING"
        )

        print(
            "=" * 65
        )

        print(
            "\nIterations :",
            self.iterations
        )

        for iteration in range(
            self.iterations
        ):

            # ==================================================
            # MATERIAL PASS
            # ==================================================

            (
                normalized_slope,
                erosion_grid,
                deposition_grid
            ) = self.material_pass()


            # ==================================================
            # STATE TRANSITION
            # ==================================================

            state_changes = (
                self.state_transition(
                    normalized_slope,
                    erosion_grid,
                    deposition_grid
                )
            )


            # ==================================================
            # ITERATION STATISTICS
            # ==================================================

            erosion = float(
                np.sum(
                    erosion_grid
                )
            )

            deposition = float(
                np.sum(
                    deposition_grid
                )
            )

            sediment_total = float(
                np.sum(
                    self.sediment
                )
            )


            # ==================================================
            # ACCUMULATE TOTALS
            # ==================================================

            self.total_erosion += (
                erosion
            )

            self.total_deposition += (
                deposition
            )


            # ==================================================
            # PRINT ITERATION RESULT
            # ==================================================

            print(
                f"Iteration "
                f"{iteration + 1:3d} | "
                f"Erosion: "
                f"{erosion:.4f} | "
                f"Deposition: "
                f"{deposition:.4f} | "
                f"Sediment: "
                f"{sediment_total:.4f} | "
                f"State Changes: "
                f"{state_changes}"
            )

    # ======================================================
    # STATISTICS
    # ======================================================

    def statistics(self):

        print(
            "\n" + "=" * 65
        )

        print(
            "R5.5 FINAL TERRAIN STATISTICS"
        )

        print(
            "=" * 65
        )


        # --------------------------------------------------
        # MATERIAL
        # --------------------------------------------------

        print(
            "\nInitial Sediment :",
            np.sum(
                self.original_sediment
            )
        )

        print(
            "Final Sediment :",
            np.sum(
                self.sediment
            )
        )

        print(
            "Total Erosion :",
            self.total_erosion
        )

        print(
            "Total Deposition :",
            self.total_deposition
        )

        print(
            "Total Transport :",
            self.total_transport
        )


        # --------------------------------------------------
        # HEIGHT CHANGE
        # --------------------------------------------------

        change = (
            self.height
            -
            self.original_height
        )

        loss = np.maximum(
            -change,
            0
        )

        gain = np.maximum(
            change,
            0
        )


        print(
            "\nFinal Height Range :",
            np.min(self.height),
            "to",
            np.max(self.height)
        )

        print(
            "Maximum Height Loss :",
            np.max(loss)
        )

        print(
            "Maximum Height Gain :",
            np.max(gain)
        )

        print(
            "Mean Absolute Height Change :",
            np.mean(
                np.abs(change)
            )
        )


        # --------------------------------------------------
        # STATES
        # --------------------------------------------------

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


        # --------------------------------------------------
        # TRANSITIONS
        # --------------------------------------------------

        print(
            "\nState Transitions:"
        )

        if not self.transition_counts:

            print(
                "No state transitions."
            )

        else:

            for (
                old,
                new
            ), count in sorted(
                self.transition_counts.items()
            ):

                print(
                    f"{old:10s} -> "
                    f"{new:10s} : "
                    f"{count}"
                )


    # ======================================================
    # SAVE
    # ======================================================

    def save(self):

        os.makedirs(
            "OUTPUT",
            exist_ok=True
        )

        np.save(
            "OUTPUT/final_height_grid.npy",
            self.height
        )

        np.save(
            "OUTPUT/final_state_grid.npy",
            self.state
        )

        np.save(
            "OUTPUT/final_sediment_grid.npy",
            self.sediment
        )

        print(
            "\nGenerated Files:"
        )

        print(
            "OUTPUT/final_height_grid.npy"
        )

        print(
            "OUTPUT/final_state_grid.npy"
        )

        print(
            "OUTPUT/final_sediment_grid.npy"
        )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 65
    )

    print(
        "CA TERRAIN - R5.5 FINAL INTEGRATION"
    )

    print(
        "=" * 65
    )


    # ======================================================
    # INPUT FILES
    # ======================================================

    height_path = os.path.join(
        "OUTPUT",
        "hydraulic_height_grid.npy"
    )

    state_path = os.path.join(
        "OUTPUT",
        "hydraulic_state_grid.npy"
    )

    sediment_path = os.path.join(
        "OUTPUT",
        "hydraulic_sediment_grid.npy"
    )


    # ======================================================
    # CHECK FILES
    # ======================================================

    for path in (
        height_path,
        state_path,
        sediment_path
    ):

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )


    # ======================================================
    # LOAD
    # ======================================================

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
        "Input Sediment :",
        np.sum(sediment_grid)
    )


    # ======================================================
    # PARAMETERS
    # ======================================================

    iterations = int(
        input(
            "\nFinal CA Iterations : "
        )
    )

    slope_threshold = float(
        input(
            "Slope Threshold : "
        )
    )

    steep_threshold = float(
        input(
            "Steep Threshold : "
        )
    )

    plateau_threshold = float(
        input(
            "Plateau Threshold : "
        )
    )

    peak_threshold = float(
        input(
            "Peak Threshold : "
        )
    )

    sediment_threshold = float(
        input(
            "Sediment Threshold : "
        )
    )


    # ======================================================
    # CREATE
    # ======================================================

    final_ca = FinalTerrainCA(

        height_grid,

        state_grid,

        sediment_grid,

        iterations=iterations,

        slope_threshold=slope_threshold,

        steep_threshold=steep_threshold,

        plateau_threshold=plateau_threshold,

        peak_threshold=peak_threshold,

        sediment_threshold=sediment_threshold
    )


    # ==================================================
    # RUN
    # ==================================================

    final_ca.run()


    # ==================================================
    # STATISTICS
    # ==================================================

    final_ca.statistics()


    # ==================================================
    # SAVE
    # ==================================================

    final_ca.save()


    print(
        "\nR5.5 FINAL INTEGRATION COMPLETED."
    )