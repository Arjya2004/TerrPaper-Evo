import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ============================================================
# VARIATION MODULE
# ============================================================

METHOD = "voronoi"   # "voronoi" or "reaction_diffusion"
SEED = 42
STRENGTH = 0.90

heightmap = np.load("rotation_heightmap.npy")

rows, cols = heightmap.shape


# ============================================================
# VORONOI
# ============================================================

def voronoi_variation(rows, cols, points=20):

    rng = np.random.default_rng(SEED)

    seeds = rng.integers(
        0, [rows, cols],
        size=(points, 2)
    )

    values = rng.uniform(
        -1, 1, points
    )

    y, x = np.indices(
        (rows, cols)
    )

    field = np.zeros(
        (rows, cols)
    )

    distance = np.full(
        (rows, cols),
        np.inf
    )

    for (sy, sx), value in zip(
        seeds, values
    ):

        d = (y - sy) ** 2 + (x - sx) ** 2

        mask = d < distance

        distance[mask] = d[mask]
        field[mask] = value

    return gaussian_filter(
        field,
        sigma=1
    )


# ============================================================
# REACTION-DIFFUSION
# ============================================================

def reaction_diffusion(
    rows,
    cols,
    iterations=300
):

    rng = np.random.default_rng(SEED)

    U = np.ones(
        (rows, cols)
    )

    V = np.zeros(
        (rows, cols)
    )

    # Initial random seeds
    for _ in range(8):

        y = rng.integers(
            2,
            rows - 2
        )

        x = rng.integers(
            2,
            cols - 2
        )

        V[
            y-1:y+2,
            x-1:x+2
        ] = 1

    for _ in range(iterations):

        lapU = (
            np.roll(U, 1, 0) +
            np.roll(U, -1, 0) +
            np.roll(U, 1, 1) +
            np.roll(U, -1, 1) -
            4 * U
        )

        lapV = (
            np.roll(V, 1, 0) +
            np.roll(V, -1, 0) +
            np.roll(V, 1, 1) +
            np.roll(V, -1, 1) -
            4 * V
        )

        reaction = U * V * V

        U += (
            0.16 * lapU
            - reaction
            + 0.035 * (1 - U)
        )

        V += (
            0.08 * lapV
            + reaction
            - (0.035 + 0.060) * V
        )

        U = np.clip(U, 0, 1)
        V = np.clip(V, 0, 1)

    V = (
        V - V.min()
    ) / (
        V.max() - V.min()
    )

    return V * 2 - 1


# ============================================================
# SELECT METHOD
# ============================================================

if METHOD == "voronoi":

    variation = voronoi_variation(
        rows,
        cols
    )

elif METHOD == "reaction_diffusion":

    variation = reaction_diffusion(
        rows,
        cols
    )

else:

    raise ValueError(
        "Unknown method"
    )


# ============================================================
# APPLY VARIATION
# ============================================================

modified = heightmap * (
    1 + STRENGTH * variation
)

modified = np.maximum(
    modified,
    0
)


# ============================================================
# SAVE
# ============================================================

np.save(
    "variation_heightmap.npy",
    modified
)

np.savetxt(
    "variation_heightmap.txt",
    modified,
    fmt="%.2f"
)


# ============================================================
# DISPLAY
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    modified,
    cmap="terrain",
    origin="lower"
)

plt.colorbar(
    label="Height"
)

plt.title(
    "Rotation + " + METHOD
)

plt.tight_layout()

plt.savefig(
    "variation_heightmap.png",
    dpi=300
)

plt.show()