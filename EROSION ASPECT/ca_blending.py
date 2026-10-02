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

from ca_rules import CARuleEngine


# ==========================================================
# PART 3
# RULE R1 - BOUNDARY BLENDING
# ==========================================================


class TerrainBlending:

    def __init__(
        self,
        height_grid,
        state_grid,
        iterations=10
    ):

        self.height = height_grid.astype(
            np.float32
        ).copy()

        self.state = state_grid.astype(
            np.uint8
        ).copy()

        self.original_height = (
            self.height.copy()
        )

        self.original_state = (
            self.state.copy()
        )

        self.iterations = iterations

        self.rows, self.cols = (
            self.height.shape
        )

        # --------------------------------------------------
        # Transition strength
        #
        # 0.0 = pure original state
        # 1.0 = fully transitioned
        # --------------------------------------------------

        self.transition_strength = np.zeros(
            self.height.shape,
            dtype=np.float32
        )

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
    # One Blending Iteration
    # ======================================================

    def apply_rule_r1(self):

        # --------------------------------------------------
        # Synchronous update buffers
        # --------------------------------------------------

        new_height = self.height.copy()

        new_state = self.state.copy()

        new_transition = (
            self.transition_strength.copy()
        )

        changed_cells = 0

        blended_cells = 0

        # --------------------------------------------------
        # Scan interior cells
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
                # R1 applies only to SOIL
                # --------------------------------------------------

                if current_state != SOIL:
                    continue

                neighbours = self.neighbours(
                    r,
                    c
                )

                slope_neighbours = 0

                neighbour_heights = []

                for nr, nc in neighbours:

                    neighbour_heights.append(
                        self.height[nr, nc]
                    )

                    if self.state[nr, nc] == SLOPE:

                        slope_neighbours += 1

                # --------------------------------------------------
                # No SLOPE influence
                # --------------------------------------------------

                if slope_neighbours == 0:

                    continue

                # --------------------------------------------------
                # SLOPE influence
                #
                # 1 neighbour = 1/8
                # 8 neighbours = 1.0
                # --------------------------------------------------

                influence = (
                    slope_neighbours / 8.0
                )

                # --------------------------------------------------
                # Increase transition strength gradually
                # --------------------------------------------------

                transition_amount = (
                    0.20 * influence
                )

                new_transition[r, c] = min(
                    1.0,
                    self.transition_strength[r, c]
                    +
                    transition_amount
                )

                blended_cells += 1

                # --------------------------------------------------
                # Height blending
                # --------------------------------------------------

                average_neighbour_height = (
                    np.mean(neighbour_heights)
                )

                current_height = (
                    self.height[r, c]
                )

                height_difference = (
                    average_neighbour_height
                    -
                    current_height
                )

                # --------------------------------------------------
                # Small gradual height adjustment
                # --------------------------------------------------

                height_change = (
                    0.15
                    *
                    influence
                    *
                    height_difference
                )

                new_height[r, c] = (
                    current_height
                    +
                    height_change
                )

                # --------------------------------------------------
                # State transition threshold
                #
                # Do NOT immediately change SOIL.
                # --------------------------------------------------

                if new_transition[r, c] >= 0.60:

                    new_state[r, c] = SLOPE

                    changed_cells += 1

        # --------------------------------------------------
        # Apply simultaneously
        # --------------------------------------------------

        self.height = new_height

        self.state = new_state

        self.transition_strength = (
            new_transition
        )

        return (
            changed_cells,
            blended_cells
        )

    # ======================================================
    # Run R1
    # ======================================================

    def run(self):

        print("\n" + "=" * 60)

        print("RULE R1 - BOUNDARY BLENDING")

        print("=" * 60)

        print(
            "\nIterations :",
            self.iterations
        )

        for iteration in range(
            self.iterations
        ):

            changed_cells, blended_cells = (
                self.apply_rule_r1()
            )

            print(
                f"Iteration "
                f"{iteration + 1:3d} | "
                f"Boundary Cells: "
                f"{blended_cells:5d} | "
                f"State Changes: "
                f"{changed_cells:5d}"
            )

        return (
            self.height,
            self.state
        )

    # ======================================================
    # Statistics
    # ======================================================

    def statistics(self):

        print("\n" + "=" * 60)

        print("BLENDING STATISTICS")

        print("=" * 60)

        original_boundary = np.sum(
            self.original_state != self.state
        )

        height_difference = (
            self.height
            -
            self.original_height
        )

        print(
            "\nChanged State Cells :",
            original_boundary
        )

        print(
            "Maximum Height Change :",
            np.max(
                np.abs(height_difference)
            )
        )

        print(
            "Average Height Change :",
            np.mean(
                np.abs(height_difference)
            )
        )

        print(
            "\nFinal Height Range :",
            np.min(self.height),
            "to",
            np.max(self.height)
        )

        print("\nFinal State Counts:")

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
    # Save Results
    # ======================================================

    def save(self):

        os.makedirs(
            "OUTPUT",
            exist_ok=True
        )

        np.save(
            "OUTPUT/blended_height_grid.npy",
            self.height
        )

        np.save(
            "OUTPUT/blended_state_grid.npy",
            self.state
        )

        np.save(
            "OUTPUT/transition_strength.npy",
            self.transition_strength
        )

        print(
            "\nGenerated Files:"
        )

        print(
            "OUTPUT/blended_height_grid.npy"
        )

        print(
            "OUTPUT/blended_state_grid.npy"
        )

        print(
            "OUTPUT/transition_strength.npy"
        )


# ==========================================================
# STANDALONE EXECUTION
# ==========================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print("CA TERRAIN BOUNDARY BLENDING")

    print("=" * 60)

    height_path = os.path.join(
        "OUTPUT",
        "height_grid.npy"
    )

    state_path = os.path.join(
        "OUTPUT",
        "state_grid.npy"
    )

    if not os.path.exists(
        height_path
    ):

        raise FileNotFoundError(
            "OUTPUT/height_grid.npy not found."
        )

    if not os.path.exists(
        state_path
    ):

        raise FileNotFoundError(
            "OUTPUT/state_grid.npy not found."
        )

    height_grid = np.load(
        height_path
    )

    state_grid = np.load(
        state_path
    )

    print(
        "\nGrid Shape :",
        height_grid.shape
    )

    print(
        "Initial Maximum Height :",
        np.max(height_grid)
    )

    print(
        "Initial Minimum Height :",
        np.min(height_grid)
    )

    # ------------------------------------------------------
    # Number of CA iterations
    # ------------------------------------------------------

    iterations = int(
        input(
            "\nBlending Iterations : "
        )
    )

    if iterations <= 0:

        raise ValueError(
            "Iterations must be greater than zero."
        )

    # ------------------------------------------------------
    # Create blending engine
    # ------------------------------------------------------

    blending = TerrainBlending(
        height_grid,
        state_grid,
        iterations=iterations
    )

    # ------------------------------------------------------
    # Run Rule R1
    # ------------------------------------------------------

    blending.run()

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    blending.statistics()

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    blending.save()

    print(
        "\nRule R1 Completed."
    )