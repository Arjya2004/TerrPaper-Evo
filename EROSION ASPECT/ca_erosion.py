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
# PART 4
# R2 - STATE-AWARE GRADIENT EROSION
# ==========================================================


class CAGradientErosion:

    def __init__(
        self,
        height_grid,
        state_grid,
        iterations=10,
        talus_threshold=1.0,
        erosion_strength=0.20,
        max_move=0.50
    ):

        self.height = (
            height_grid
            .astype(np.float32)
            .copy()
        )

        self.state = (
            state_grid
            .astype(np.uint8)
            .copy()
        )

        self.original_height = (
            self.height.copy()
        )

        self.original_state = (
            self.state.copy()
        )

        self.iterations = iterations

        self.talus_threshold = (
            talus_threshold
        )

        self.erosion_strength = (
            erosion_strength
        )

        self.max_move = max_move

        self.rows, self.cols = (
            self.height.shape
        )

        # --------------------------------------------------
        # Sediment produced by erosion
        # --------------------------------------------------

        self.sediment = np.zeros(
            self.height.shape,
            dtype=np.float32
        )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        self.total_eroded = 0.0

        self.active_cells = 0

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

                result.append(
                    (nr, nc)
                )

        return result

    # ======================================================
    # State-dependent erosion resistance
    # ======================================================

    def erosion_factor(self, state):

        if state == SOIL:

            return 0.35

        if state == SLOPE:

            return 0.25

        if state == ROCK:

            return 0.10

        if state == PLATEAU:

            return 0.05

        if state == PEAK:

            return 0.02

        return 0.0

    # ======================================================
    # State-dependent threshold
    # ======================================================

    def state_threshold(self, state):

        if state == SOIL:

            return self.talus_threshold * 0.75

        if state == SLOPE:

            return self.talus_threshold

        if state == ROCK:

            return self.talus_threshold * 1.50

        if state == PLATEAU:

            return self.talus_threshold * 1.75

        if state == PEAK:

            return self.talus_threshold * 2.00

        return 999999.0

    # ======================================================
    # Find downhill neighbours
    # ======================================================

    def downhill_neighbours(
        self,
        r,
        c
    ):

        current_height = (
            self.height[r, c]
        )

        downhill = []

        for nr, nc in self.neighbours(
            r,
            c
        ):

            neighbour_height = (
                self.height[nr, nc]
            )

            drop = (
                current_height
                -
                neighbour_height
            )

            if drop > 0:

                downhill.append(
                    (
                        nr,
                        nc,
                        drop
                    )
                )

        return downhill

    # ======================================================
    # R2 - One Synchronous Iteration
    # ======================================================

    def apply_rule_r2(self):

        # --------------------------------------------------
        # Synchronous height update
        # --------------------------------------------------

        delta = np.zeros(
            self.height.shape,
            dtype=np.float32
        )

        # --------------------------------------------------
        # Sediment generated during this iteration
        # --------------------------------------------------

        sediment_delta = np.zeros(
            self.height.shape,
            dtype=np.float32
        )

        # --------------------------------------------------
        # State changes are applied after scanning
        # --------------------------------------------------

        new_state = self.state.copy()

        iteration_eroded = 0.0

        iteration_active = 0

        iteration_state_changes = 0

        # --------------------------------------------------
        # Interior cells
        # --------------------------------------------------

        for r in range(
            1,
            self.rows - 1
        ):

            for c in range(
                1,
                self.cols - 1
            ):

                current_state = int(
                    self.state[r, c]
                )

                # --------------------------------------------------
                # Only terrain material can erode
                # --------------------------------------------------

                if current_state not in (
                    SOIL,
                    SLOPE,
                    ROCK,
                    PLATEAU,
                    PEAK
                ):

                    continue

                current_height = float(
                    self.height[r, c]
                )

                downhill = (
                    self.downhill_neighbours(
                        r,
                        c
                    )
                )

                if len(downhill) == 0:

                    continue

                # --------------------------------------------------
                # Strongest downhill direction
                # --------------------------------------------------

                steepest = max(
                    downhill,
                    key=lambda x: x[2]
                )

                nr, nc, largest_drop = (
                    steepest
                )

                threshold = (
                    self.state_threshold(
                        current_state
                    )
                )

                # --------------------------------------------------
                # R2 erosion condition
                # --------------------------------------------------

                excess = (
                    largest_drop
                    -
                    threshold
                )

                if excess <= 0:

                    continue

                iteration_active += 1

                # --------------------------------------------------
                # State-dependent resistance
                # --------------------------------------------------

                resistance = (
                    self.erosion_factor(
                        current_state
                    )
                )

                # --------------------------------------------------
                # Calculate erosion amount
                # --------------------------------------------------

                amount = (
                    excess
                    *
                    self.erosion_strength
                    *
                    resistance
                )

                amount = min(
                    amount,
                    self.max_move
                )

                # --------------------------------------------------
                # Safety condition
                # --------------------------------------------------

                if amount <= 0:

                    continue

                # Never remove more than available height
                amount = min(
                    amount,
                    max(
                        0.0,
                        current_height
                        -
                        0.1
                    )
                )

                if amount <= 0:

                    continue

                # --------------------------------------------------
                # Remove material
                # --------------------------------------------------

                delta[r, c] -= amount

                # --------------------------------------------------
                # Store sediment
                # --------------------------------------------------

                sediment_delta[r, c] += (
                    amount
                )

                iteration_eroded += amount

                # --------------------------------------------------
                # State transition
                #
                # Strong erosion can convert:
                #
                # SOIL  -> SEDIMENT
                # SLOPE -> SEDIMENT
                #
                # ROCK / PLATEAU / PEAK remain
                # structurally protected.
                # --------------------------------------------------

                if current_state in (
                    SOIL,
                    SLOPE
                ):

                    erosion_ratio = (
                        amount
                        /
                        max(
                            current_height,
                            1.0
                        )
                    )

                    if erosion_ratio >= 0.08:

                        new_state[
                            r, c
                        ] = SEDIMENT

                        iteration_state_changes += 1

        # --------------------------------------------------
        # Apply height changes simultaneously
        # --------------------------------------------------

        self.height += delta

        # --------------------------------------------------
        # Apply sediment changes
        # --------------------------------------------------

        self.sediment += (
            sediment_delta
        )

        # --------------------------------------------------
        # Apply state changes
        # --------------------------------------------------

        self.state = new_state

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        self.total_eroded += (
            iteration_eroded
        )

        self.active_cells += (
            iteration_active
        )

        self.state_changes += (
            iteration_state_changes
        )

        return (
            iteration_active,
            iteration_eroded,
            iteration_state_changes
        )

    # ======================================================
    # Run R2
    # ======================================================

    def run(self):

        print("\n" + "=" * 60)

        print(
            "R2 - STATE-AWARE GRADIENT EROSION"
        )

        print("=" * 60)

        print(
            "\nIterations :",
            self.iterations
        )

        print(
            "Talus Threshold :",
            self.talus_threshold
        )

        print(
            "Erosion Strength :",
            self.erosion_strength
        )

        print(
            "Maximum Move :",
            self.max_move
        )

        for iteration in range(
            self.iterations
        ):

            (
                active,
                eroded,
                changes
            ) = self.apply_rule_r2()

            print(
                f"Iteration "
                f"{iteration + 1:3d} | "
                f"Active Cells: "
                f"{active:5d} | "
                f"Eroded: "
                f"{eroded:.4f} | "
                f"State Changes: "
                f"{changes:4d}"
            )

        return (
            self.height,
            self.state,
            self.sediment
        )

    # ======================================================
    # Statistics
    # ======================================================

    def statistics(self):

        print("\n" + "=" * 60)

        print(
            "R2 EROSION STATISTICS"
        )

        print("=" * 60)

        height_change = (
            self.height
            -
            self.original_height
        )

        print(
            "\nTotal Eroded Material :",
            self.total_eroded
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
            "Total Sediment :",
            np.sum(
                self.sediment
            )
        )

        print(
            "Total Active Cells :",
            self.active_cells
        )

        print(
            "State Changes :",
            self.state_changes
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
            "OUTPUT/eroded_height_grid.npy",
            self.height
        )

        np.save(
            "OUTPUT/eroded_state_grid.npy",
            self.state
        )

        np.save(
            "OUTPUT/sediment_grid.npy",
            self.sediment
        )

        print(
            "\nGenerated Files:"
        )

        print(
            "OUTPUT/eroded_height_grid.npy"
        )

        print(
            "OUTPUT/eroded_state_grid.npy"
        )

        print(
            "OUTPUT/sediment_grid.npy"
        )


# ==========================================================
# STANDALONE EXECUTION
# ==========================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        "CA TERRAIN - R2 EROSION"
    )

    print("=" * 60)

    height_path = os.path.join(
        "OUTPUT",
        "blended_height_grid.npy"
    )

    state_path = os.path.join(
        "OUTPUT",
        "blended_state_grid.npy"
    )

    # ------------------------------------------------------
    # Check input files
    # ------------------------------------------------------

    if not os.path.exists(
        height_path
    ):

        raise FileNotFoundError(
            "Run ca_blending.py first."
        )

    if not os.path.exists(
        state_path
    ):

        raise FileNotFoundError(
            "Run ca_blending.py first."
        )

    # ------------------------------------------------------
    # Load R1 output
    # ------------------------------------------------------

    height_grid = np.load(
        height_path
    )

    state_grid = np.load(
        state_path
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

    # ------------------------------------------------------
    # User parameters
    # ------------------------------------------------------

    iterations = int(
        input(
            "\nErosion Iterations : "
        )
    )

    if iterations <= 0:

        raise ValueError(
            "Iterations must be greater than zero."
        )

    talus_threshold = float(
        input(
            "Talus Threshold : "
        )
    )

    if talus_threshold <= 0:

        raise ValueError(
            "Talus Threshold must be greater than zero."
        )

    erosion_strength = float(
        input(
            "Erosion Strength (0-1) : "
        )
    )

    if not (
        0 < erosion_strength <= 1
    ):

        raise ValueError(
            "Erosion Strength must be between 0 and 1."
        )

    max_move = float(
        input(
            "Maximum Material Move : "
        )
    )

    if max_move <= 0:

        raise ValueError(
            "Maximum Material Move must be greater than zero."
        )

    # ------------------------------------------------------
    # Create R2 engine
    # ------------------------------------------------------

    erosion = CAGradientErosion(

        height_grid,

        state_grid,

        iterations=iterations,

        talus_threshold=talus_threshold,

        erosion_strength=erosion_strength,

        max_move=max_move
    )

    # ------------------------------------------------------
    # Run
    # ------------------------------------------------------

    erosion.run()

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    erosion.statistics()

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    erosion.save()

    print(
        "\nR2 Erosion Completed."
    )