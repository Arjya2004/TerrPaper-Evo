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
# CA RULE ENGINE - PART 2
# NEIGHBOURHOOD ANALYZER
# ==========================================================


class CARuleEngine:

    def __init__(self, height_grid, state_grid):

        self.height = height_grid.astype(np.float32)

        self.state = state_grid.astype(np.uint8)

        self.rows, self.cols = self.height.shape

    # ======================================================
    # Moore Neighbourhood
    # ======================================================

    def get_neighbours(self, r, c):

        neighbours = []

        for dr in (-1, 0, 1):

            for dc in (-1, 0, 1):

                # Skip current cell
                if dr == 0 and dc == 0:
                    continue

                nr = r + dr
                nc = c + dc

                # Ignore outside grid
                if nr < 0 or nr >= self.rows:
                    continue

                if nc < 0 or nc >= self.cols:
                    continue

                neighbours.append((nr, nc))

        return neighbours

    # ======================================================
    # Analyse One Cell
    # ======================================================

    def analyse_cell(self, r, c):

        current_state = int(
            self.state[r, c]
        )

        current_height = float(
            self.height[r, c]
        )

        neighbours = self.get_neighbours(r, c)

        neighbour_states = []

        neighbour_heights = []

        for nr, nc in neighbours:

            neighbour_states.append(
                int(self.state[nr, nc])
            )

            neighbour_heights.append(
                float(self.height[nr, nc])
            )

        # --------------------------------------------------
        # State Counts
        # --------------------------------------------------

        state_counts = {

            state: neighbour_states.count(state)

            for state in range(8)

        }

        # --------------------------------------------------
        # Height Statistics
        # --------------------------------------------------

        if neighbour_heights:

            average_height = np.mean(
                neighbour_heights
            )

            minimum_height = np.min(
                neighbour_heights
            )

            maximum_height = np.max(
                neighbour_heights
            )

        else:

            average_height = current_height

            minimum_height = current_height

            maximum_height = current_height

        # --------------------------------------------------
        # Height Difference
        # --------------------------------------------------

        height_difference = (

            current_height
            -
            average_height

        )

        # --------------------------------------------------
        # Local Slope
        # --------------------------------------------------

        local_slope = np.mean([

            abs(
                current_height - nh
            )

            for nh in neighbour_heights

        ]) if neighbour_heights else 0.0

        # --------------------------------------------------
        # Dominant Neighbour State
        # --------------------------------------------------

        dominant_state = max(

            state_counts,

            key=state_counts.get

        )

        dominant_count = state_counts[
            dominant_state
        ]

        return {

            "current_state":
                current_state,

            "current_state_name":
                STATE_NAMES[current_state],

            "current_height":
                current_height,

            "neighbour_count":
                len(neighbours),

            "state_counts":
                state_counts,

            "average_height":
                float(average_height),

            "minimum_height":
                float(minimum_height),

            "maximum_height":
                float(maximum_height),

            "height_difference":
                float(height_difference),

            "local_slope":
                float(local_slope),

            "dominant_state":
                dominant_state,

            "dominant_state_name":
                STATE_NAMES[dominant_state],

            "dominant_count":
                dominant_count
        }

    # ======================================================
    # Analyse Entire Grid
    # ======================================================

    def analyse_grid(self):

        print("\n" + "=" * 60)

        print("CA NEIGHBOURHOOD ANALYSIS")

        print("=" * 60)

        boundary_cells = 0

        homogeneous_cells = 0

        transition_cells = 0

        for r in range(

            1,

            self.rows - 1

        ):

            for c in range(

                1,

                self.cols - 1

            ):

                result = self.analyse_cell(

                    r,

                    c

                )

                counts = result[
                    "state_counts"
                ]

                current = result[
                    "current_state"
                ]

                # ------------------------------------------
                # Number of different neighbour states
                # ------------------------------------------

                active_states = sum(

                    1

                    for count in counts.values()

                    if count > 0

                )

                # ------------------------------------------
                # Boundary / transition cell
                # ------------------------------------------

                if active_states >= 2:

                    transition_cells += 1

                else:

                    homogeneous_cells += 1

                # ------------------------------------------
                # State boundary
                # ------------------------------------------

                if (

                    result["dominant_state"]

                    !=

                    current

                ):

                    boundary_cells += 1

        print(

            "Homogeneous Cells :",

            homogeneous_cells

        )

        print(

            "Transition Cells  :",

            transition_cells

        )

        print(

            "Boundary Cells    :",

            boundary_cells

        )

    # ======================================================
    # Inspect One Cell
    # ======================================================

    def inspect_cell(self, r, c):

        result = self.analyse_cell(

            r,

            c

        )

        print("\n" + "=" * 60)

        print(

            f"CELL ({r}, {c})"

        )

        print("=" * 60)

        print(

            "Current State :",

            result["current_state"],

            "-",

            result["current_state_name"]

        )

        print(

            "Current Height :",

            round(

                result["current_height"],

                3

            )

        )

        print(

            "Average Neighbour Height :",

            round(

                result["average_height"],

                3

            )

        )

        print(

            "Height Difference :",

            round(

                result["height_difference"],

                3

            )

        )

        print(

            "Local Slope :",

            round(

                result["local_slope"],

                3

            )

        )

        print(

            "\nNeighbour State Counts:"

        )

        for state in range(8):

            count = result[
                "state_counts"
            ][state]

            if count > 0:

                print(

                    f"  {state} - "

                    f"{STATE_NAMES[state]} : "

                    f"{count}"

                )

        print(

            "\nDominant Neighbour :",

            result["dominant_state"],

            "-",

            result["dominant_state_name"]

        )

        print(

            "Dominant Count :",

            result["dominant_count"]

        )

        return result
            # ======================================================
    # Find Transition Cells
    # ======================================================

    def find_transition_cells(self, limit=10):

        results = []

        for r in range(1, self.rows - 1):

            for c in range(1, self.cols - 1):

                current = int(
                    self.state[r, c]
                )

                neighbours = self.get_neighbours(
                    r,
                    c
                )

                neighbour_states = [

                    int(self.state[nr, nc])

                    for nr, nc in neighbours

                ]

                different = [

                    s

                    for s in neighbour_states

                    if s != current

                ]

                if len(different) > 0:

                    results.append(
                        (r, c, current, len(different))
                    )

                    if len(results) >= limit:

                        return results

        return results


if __name__ == "__main__":

    print("\n" + "=" * 60)

    print("CA NEIGHBOURHOOD ANALYZER")

    print("=" * 60)

    height_path = os.path.join(
        "OUTPUT",
        "height_grid.npy"
    )

    state_path = os.path.join(
        "OUTPUT",
        "state_grid.npy"
    )

    if not os.path.exists(height_path):

        raise FileNotFoundError(
            "OUTPUT/height_grid.npy not found."
        )

    if not os.path.exists(state_path):

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

    engine = CARuleEngine(
        height_grid,
        state_grid
    )

    # ------------------------------------------------------
    # Whole grid analysis
    # ------------------------------------------------------

    engine.analyse_grid()

    # ------------------------------------------------------
    # Find transition cells
    # ------------------------------------------------------

    print("\n" + "=" * 60)

    print("TRANSITION CELL SAMPLES")

    print("=" * 60)

    transition_cells = engine.find_transition_cells(
        limit=10
    )

    for r, c, state, different in transition_cells:

        print(
            f"Cell ({r}, {c}) | "
            f"State: {state} - "
            f"{STATE_NAMES[state]} | "
            f"Different Neighbours: {different}"
        )

    # ------------------------------------------------------
    # Inspect transition cells
    # ------------------------------------------------------

    print("\n" + "=" * 60)

    print("DETAILED TRANSITION CELLS")

    print("=" * 60)

    for r, c, state, different in transition_cells[:3]:

        engine.inspect_cell(
            r,
            c
        )

    print(
        "\nNeighbourhood Analysis Completed."
    )