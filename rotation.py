import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# LOAD VERTICAL PROFILE
# =====================================================

profile = np.load("vertical_profile.npy")

rows, cols = profile.shape

print("=" * 50)
print("ROTATIONAL SWEEP")
print("=" * 50)
print("Profile Shape :", profile.shape)

# =====================================================
# CREATE REVOLUTION VOLUME
# =====================================================

radius = cols // 2
size = radius * 2 + 1

voxels = np.zeros((size, size, rows), dtype=bool)

cx = radius
cy = radius

# =====================================================
# SOLID ROTATIONAL SWEEP
# =====================================================

for z in range(rows):

    row = profile[z]

    for x in range(cols):

        if row[x] == 0:
            continue

        # Distance from rotation axis
        r = abs(x - cols // 2)

        # Create a SOLID circular disk
        for i in range(size):
            for j in range(size):

                distance = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)

                if distance <= r:
                    voxels[i, j, z] = True

# =====================================================
# GENERATE REUSABLE HEIGHTMAP PATCH
# =====================================================

heightmap = np.sum(voxels, axis=2)

np.save("rotation_heightmap.npy", heightmap)

np.savetxt(
    "rotation_heightmap.txt",
    heightmap,
    fmt="%d"
)

plt.figure(figsize=(6,6))

plt.imshow(
    heightmap,
    cmap="terrain",
    origin="lower"
)

plt.colorbar(label="Height")

plt.title("Rotation Heightmap")

plt.xlabel("X")

plt.ylabel("Y")

plt.tight_layout()

plt.savefig(
    "rotation_heightmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Rotation Heightmap Saved Successfully.")

# =====================================================
# DISPLAY 3D ROTATED TERRAIN
# =====================================================

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection="3d")

ax.voxels(
    voxels,
    facecolors="green",
    edgecolor="black"
)

ax.set_title("Rotational Sweep of Vertical Profile")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Generation")

plt.tight_layout()

plt.show()