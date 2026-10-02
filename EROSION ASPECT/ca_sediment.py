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
# PART 5
# R3 - SEDIMENT TRANSPORT + DEPOSITION
# ==========================================================


class CASedimentTransport:

    def __init__(
        self,
        height_grid,
        state_grid,
        sediment_grid,
        iterations=10,
        transport_rate=0.60,
        deposition_rate=0.35,
        transport_threshold=0.20,
        deposition_threshold=0.50,
        max_deposit=0.50
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

        self.original_height = self.height.copy()

        self.original_sediment = self.sediment.copy()

        self.iterations = iterations

        self.transport_rate = transport_rate

        self.deposition_rate = deposition_rate

        self.transport_threshold = (
            transport_threshold
        )

        self.deposition_threshold = (
            deposition_threshold
        )

        self.max_deposit = max_deposit

        self.rows, self.cols = self.height.shape

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        self.total_transported = 0.0

        self.total_deposited = 0.0

        self.transport_events = 0

        self.deposition_events = 0

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
    # Find downhill neighbours
    # ======================================================

    def downhill_neighbours(self, r, c):

        current_height = self.height[r, c]

        downhill = []

        for nr, nc in self.neighbours(r, c):

            neighbour_height = self.height[nr, nc]

            drop = (
                current_height
                -
                neighbour_height
            )

            if drop > self.transport_threshold:

                downhill.append(
                    (nr, nc, drop)
                )

        return downhill

    # ======================================================
    # R3 - One Synchronous Iteration
    # ======================================================

    def apply_rule_r3(self):

        # --------------------------------------------------
        # Changes are accumulated first
        # --------------------------------------------------

        sediment_delta = np.zeros(
            self.sediment.shape,
            dtype=np.float32
        )

        height_delta = np.zeros(
            self.height.shape,
            dtype=np.float32
        )

        state_changes = []

        iteration_transport = 0.0

        iteration_deposit = 0.0

        iteration_transport_events = 0

        iteration_deposition_events = 0

        # --------------------------------------------------
        # PASS 1
        # Determine sediment movement
        # --------------------------------------------------

        for r in range(1, self.rows - 1):

            for c in range(1, self.cols - 1):

                available = float(
                    self.sediment[r, c]
                )

                if available <= 0:

                    continue

                downhill = (
                    self.downhill_neighbours(
                        r,
                        c
                    )
                )

                # --------------------------------------------------
                # No downhill route
                # --------------------------------------------------

                if len(downhill) == 0:

                    continue

                # --------------------------------------------------
                # Sort by steepness
                # --------------------------------------------------

                downhill.sort(
                    key=lambda x: x[2],
                    reverse=True
                )

                # --------------------------------------------------
                # Only a fraction moves per iteration
                # --------------------------------------------------

                movable = (
                    available
                    *
                    self.transport_rate
                )

                # --------------------------------------------------
                # Limit movement
                # --------------------------------------------------

                movable = min(
                    movable,
                    available
                )

                if movable <= 0:

                    continue

                # --------------------------------------------------
                # Distribute according to height drop
                # --------------------------------------------------

                total_drop = sum(
                    drop
                    for _, _, drop in downhill
                )

                if total_drop <= 0:

                    continue

                moved_here = 0.0

                for nr, nc, drop in downhill:

                    fraction = (
                        drop
                        /
                        total_drop
                    )

                    amount = (
                        movable
                        *
                        fraction
                    )

                    if amount <= 0:

                        continue

                    sediment_delta[r, c] -= amount

                    sediment_delta[nr, nc] += amount

                    moved_here += amount

                if moved_here > 0:

                    iteration_transport += (
                        moved_here
                    )

                    iteration_transport_events += 1

        # --------------------------------------------------
        # APPLY TRANSPORT
        # --------------------------------------------------

        self.sediment += sediment_delta

        # --------------------------------------------------
        # PASS 2
        # Deposition
        # --------------------------------------------------

        deposition_delta = np.zeros(
            self.sediment.shape,
            dtype=np.float32
        )

        for r in range(1, self.rows - 1):

            for c in range(1, self.cols - 1):

                available = float(
                    self.sediment[r, c]
                )

                if available <= 0:

                    continue

                downhill = (
                    self.downhill_neighbours(
                        r,
                        c
                    )
                )

                # --------------------------------------------------
                # No sufficiently lower neighbour
                # means sediment can settle.
                # --------------------------------------------------

                if len(downhill) == 0:

                    deposit_amount = (
                        available
                        *
                        self.deposition_rate
                    )

                    deposit_amount = min(
                        deposit_amount,
                        self.max_deposit
                    )

                    if deposit_amount <= 0:

                        continue

                    deposition_delta[r, c] -= (
                        deposit_amount
                    )

                    height_delta[r, c] += (
                        deposit_amount
                    )

                    iteration_deposit += (
                        deposit_amount
                    )

                    iteration_deposition_events += 1

                    state_changes.append(
                        (r, c, SOIL)
                    )

        # --------------------------------------------------
        # Apply deposition
        # --------------------------------------------------

        self.sediment += deposition_delta

        self.height += height_delta

        # --------------------------------------------------
        # Update states
        # --------------------------------------------------

        for r, c, new_state in state_changes:

            self.state[r, c] = new_state

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        self.total_transported += (
            iteration_transport
        )

        self.total_deposited += (
            iteration_deposit
        )

        self.transport_events += (
            iteration_transport_events
        )

        self.deposition_events += (
            iteration_deposition_events
        )

        return (
            iteration_transport,
            iteration_deposit,
            iteration_transport_events,
            iteration_deposition_events
        )

    # ======================================================
    # Run R3
    # ======================================================

    def run(self):

        print("\n" + "=" * 60)

        print(
            "R3 - SEDIMENT TRANSPORT + DEPOSITION"
        )

        print("=" * 60)

        print(
            "\nIterations :",
            self.iterations
        )

        print(
            "Transport Rate :",
            self.transport_rate
        )

        print(
            "Deposition Rate :",
            self.deposition_rate
        )

        print(
            "Transport Threshold :",
            self.transport_threshold
        )

        print(
            "Deposition Threshold :",
            self.deposition_threshold
        )

        for iteration in range(
            self.iterations
        ):

            (
                transported,
                deposited,
                transport_events,
                deposition_events
            ) = self.apply_rule_r3()

            print(
                f"Iteration "
                f"{iteration + 1:3d} | "
                f"Transported: "
                f"{transported:.4f} | "
                f"Deposited: "
                f"{deposited:.4f} | "
                f"Transport Events: "
                f"{transport_events:5d} | "
                f"Deposition Events: "
                f"{deposition_events:5d}"
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
            "R3 SEDIMENT STATISTICS"
        )

        print("=" * 60)

        height_change = (
            self.height
            -
            self.original_height
        )

        initial_sediment = np.sum(
            self.original_sediment
        )

        final_sediment = np.sum(
            self.sediment
        )

        print(
            "\nInitial Sediment :",
            initial_sediment
        )

        print(
            "Final Sediment :",
            final_sediment
        )

        print(
            "Total Transported :",
            self.total_transported
        )

        print(
            "Total Deposited :",
            self.total_deposited
        )

        print(
            "Transport Events :",
            self.transport_events
        )

        print(
            "Deposition Events :",
            self.deposition_events
        )

        print(
            "\nMaximum Height Gain :",
            np.max(height_change)
        )

        print(
            "Maximum Height Loss :",
            abs(np.min(height_change))
        )

        print(
            "Mean Absolute Height Change :",
            np.mean(
                np.abs(height_change)
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
            "OUTPUT/transported_height_grid.npy",
            self.height
        )

        np.save(
            "OUTPUT/transported_state_grid.npy",
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
            "OUTPUT/transported_height_grid.npy"
        )

        print(
            "OUTPUT/transported_state_grid.npy"
        )

        print(
            "OUTPUT/final_sediment_grid.npy"
        )


# ==========================================================
# STANDALONE EXECUTION
# ==========================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        "CA TERRAIN - R3 SEDIMENT TRANSPORT"
    )

    print("=" * 60)

    height_path = os.path.join(
        "OUTPUT",
        "eroded_height_grid.npy"
    )

    state_path = os.path.join(
        "OUTPUT",
        "eroded_state_grid.npy"
    )

    sediment_path = os.path.join(
        "OUTPUT",
        "sediment_grid.npy"
    )

    # ------------------------------------------------------
    # Check inputs
    # ------------------------------------------------------

    if not os.path.exists(
        height_path
    ):

        raise FileNotFoundError(
            "Run ca_erosion.py first."
        )

    if not os.path.exists(
        state_path
    ):

        raise FileNotFoundError(
            "Run ca_erosion.py first."
        )

    if not os.path.exists(
        sediment_path
    ):

        raise FileNotFoundError(
            "Run ca_erosion.py first."
        )

    # ------------------------------------------------------
    # Load R2 output
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
            "\nTransport Iterations : "
        )
    )

    if iterations <= 0:

        raise ValueError(
            "Iterations must be greater than zero."
        )

    transport_rate = float(
        input(
            "Transport Rate (0-1) : "
        )
    )

    if not (
        0 < transport_rate <= 1
    ):

        raise ValueError(
            "Transport Rate must be between 0 and 1."
        )

    deposition_rate = float(
        input(
            "Deposition Rate (0-1) : "
        )
    )

    if not (
        0 < deposition_rate <= 1
    ):

        raise ValueError(
            "Deposition Rate must be between 0 and 1."
        )

    transport_threshold = float(
        input(
            "Transport Threshold : "
        )
    )

    if transport_threshold < 0:

        raise ValueError(
            "Transport Threshold cannot be negative."
        )

    deposition_threshold = float(
        input(
            "Deposition Threshold : "
        )
    )

    if deposition_threshold < 0:

        raise ValueError(
            "Deposition Threshold cannot be negative."
        )

    max_deposit = float(
        input(
            "Maximum Deposit : "
        )
    )

    if max_deposit <= 0:

        raise ValueError(
            "Maximum Deposit must be greater than zero."
        )

    # ------------------------------------------------------
    # Create R3 engine
    # ------------------------------------------------------

    transport = CASedimentTransport(

        height_grid,

        state_grid,

        sediment_grid,

        iterations=iterations,

        transport_rate=transport_rate,

        deposition_rate=deposition_rate,

        transport_threshold=transport_threshold,

        deposition_threshold=deposition_threshold,

        max_deposit=max_deposit
    )

    # ------------------------------------------------------
    # Run
    # ------------------------------------------------------

    transport.run()

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    transport.statistics()

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    transport.save()

    print(
        "\nR3 Sediment Transport Completed."
    )