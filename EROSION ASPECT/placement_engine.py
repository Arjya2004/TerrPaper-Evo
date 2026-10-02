import os
import json
import numpy as np

# ==========================================================
# TERRAIN TYPE IDs
# ==========================================================

EMPTY = 0
MOUNTAIN = 1
VOLCANO = 2
PLATEAU = 3
HILL = 4
RIVER = 5
CANYON = 6
CUSTOM = 7

FIXED = 9


# ==========================================================
# TERRAIN DATA
# ==========================================================

class TerrainData:

    def __init__(

        self,

        terrain_size,

        ground_height=0

    ):

        self.size = terrain_size

        self.ground = ground_height

        # ------------------------------------------
        # Height Grid
        # ------------------------------------------

        self.height_grid = np.full(

            (terrain_size, terrain_size),

            ground_height,

            dtype=np.float32

        )

        # ------------------------------------------
        # Terrain State
        # ------------------------------------------

        self.state_grid = np.zeros(

            (terrain_size, terrain_size),

            dtype=np.uint8

        )

        # ------------------------------------------
        # Terrain Type
        # ------------------------------------------

        self.type_grid = np.zeros(

            (terrain_size, terrain_size),

            dtype=np.uint8

        )

        # ------------------------------------------
        # Fixed Mask
        # ------------------------------------------

        self.fixed_mask = np.zeros(

            (terrain_size, terrain_size),

            dtype=bool

        )

        # ------------------------------------------
        # Placement History
        # ------------------------------------------

        self.placements = []


# ==========================================================
# HEIGHTMAP LIBRARY
# ==========================================================

class PrimitiveLibrary:

    def __init__(

        self,

        folder

    ):

        self.folder = folder

        self.heightmaps = {}

    # ------------------------------------------------------

    def load(self):

        if not os.path.exists(self.folder):

            raise FileNotFoundError(

                f"Folder '{self.folder}' not found."

            )

        files = sorted(

            os.listdir(self.folder)

        )

        for file in files:

            if not file.endswith(".npy"):

                continue

            name = file[:-4]

            path = os.path.join(

                self.folder,

                file

            )

            self.heightmaps[name] = np.load(

                path

            ).astype(

                np.float32

            )

        print("\nLoaded Heightmaps:\n")

        for i, name in enumerate(self.heightmaps.keys()):

            print(f"{i+1}. {name}")

    # ------------------------------------------------------

    def names(self):

        return list(

            self.heightmaps.keys()

        )

    # ------------------------------------------------------

    def get(

        self,

        name

    ):

        return self.heightmaps[name]
    # ==========================================================
# PLACEMENT ENGINE
# ==========================================================

class PlacementEngine:

    def __init__(

        self,

        terrain_data,

        primitive_library

    ):

        self.data = terrain_data

        self.library = primitive_library

    # ------------------------------------------------------
    # Check Grid Boundary
    # ------------------------------------------------------

    def inside(

        self,

        r,

        c

    ):

        return (

            0 <= r < self.data.size

            and

            0 <= c < self.data.size

        )

    # ------------------------------------------------------
    # Automatically Detect Terrain Type
    # ------------------------------------------------------

    def detect_type(

        self,

        patch_name

    ):

        name = patch_name.upper()

        if "MOUNTAIN" in name:

            return MOUNTAIN

        elif "VOLCANO" in name:

            return VOLCANO

        elif "PLATEAU" in name:

            return PLATEAU

        elif "HILL" in name:

            return HILL

        elif "RIVER" in name:

            return RIVER

        elif "CANYON" in name:

            return CANYON

        else:

            return CUSTOM

    # ------------------------------------------------------
    # Place Heightmap
    # ------------------------------------------------------

    def place_patch(

        self,

        patch_name,

        center_x,

        center_y

    ):

        patch = self.library.get(

            patch_name

        )

        terrain_type = self.detect_type(

            patch_name

        )

        rows, cols = patch.shape

        half_r = rows // 2

        half_c = cols // 2

        for i in range(rows):

            for j in range(cols):

                value = patch[i, j]

                if value <= 0:

                    continue

                x = center_x - half_r + i

                y = center_y - half_c + j

                if not self.inside(

                    x,

                    y

                ):

                    continue

                new_height = (

                    self.data.ground

                    +

                    value

                )

                self.data.height_grid[x, y] = max(

                    self.data.height_grid[x, y],

                    new_height

                )

                self.data.state_grid[x, y] = FIXED

                self.data.fixed_mask[x, y] = True

                self.data.type_grid[x, y] = terrain_type

        self.data.placements.append(

            {

                "primitive": patch_name,

                "x": center_x,

                "y": center_y,

                "terrain_type": terrain_type

            }

        )

        print(

            f"\n{patch_name} placed successfully."

        )
            # ------------------------------------------------------
    # Interactive Placement
    # ------------------------------------------------------

    def interactive_place(self):

        names = self.library.names()

        while True:

            print("\n" + "=" * 60)

            print("AVAILABLE HEIGHTMAPS")

            print("=" * 60)

            for i, name in enumerate(names):

                print(f"{i+1}. {name}")

            print("\nQ. Finish Placement")

            choice = input(

                "\nChoose Heightmap : "

            ).strip()

            # ------------------------------------------
            # Finish
            # ------------------------------------------

            if choice.upper() == "Q":

                break

            # ------------------------------------------
            # Validate Input
            # ------------------------------------------

            try:

                choice = int(choice)

            except ValueError:

                print("\nInvalid Input.")

                continue

            if choice < 1 or choice > len(names):

                print("\nInvalid Heightmap Number.")

                continue

            patch_name = names[choice - 1]

            # ------------------------------------------
            # Position
            # ------------------------------------------

            try:

                center_x = int(

                    input(

                        "Center X : "

                    )

                )

                center_y = int(

                    input(

                        "Center Y : "

                    )

                )

            except ValueError:

                print("\nInvalid Coordinates.")

                continue

            # ------------------------------------------
            # Place Terrain
            # ------------------------------------------

            self.place_patch(

                patch_name,

                center_x,

                center_y

            )

            print("\nTerrain Added Successfully.")

        print("\nPlacement Finished.")
            # ------------------------------------------------------
    # Save Terrain
    # ------------------------------------------------------

    def save_all(

        self,

        output_folder="OUTPUT"

    ):

        os.makedirs(

            output_folder,

            exist_ok=True

        )

        np.save(

            os.path.join(

                output_folder,

                "height_grid.npy"

            ),

            self.data.height_grid

        )

        np.save(

            os.path.join(

                output_folder,

                "terrain_state.npy"

            ),

            self.data.state_grid

        )

        np.save(

            os.path.join(

                output_folder,

                "terrain_type.npy"

            ),

            self.data.type_grid

        )

        np.save(

            os.path.join(

                output_folder,

                "fixed_mask.npy"

            ),

            self.data.fixed_mask

        )

        with open(

            os.path.join(

                output_folder,

                "placement_info.json"

            ),

            "w"

        ) as file:

            json.dump(

                self.data.placements,

                file,

                indent=4

            )

        print("\n========================================")

        print("Terrain Saved Successfully")

        print("========================================")

        print("OUTPUT/height_grid.npy")

        print("OUTPUT/terrain_state.npy")

        print("OUTPUT/terrain_type.npy")

        print("OUTPUT/fixed_mask.npy")

        print("OUTPUT/placement_info.json")


    # ------------------------------------------------------
    # Run
    # ------------------------------------------------------

    def run(self):

        self.interactive_place()

        self.save_all()

        return (

            self.data.height_grid,

            self.data.state_grid,

            self.data.fixed_mask,

            self.data.type_grid

        )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)

    print("INTERACTIVE TERRAIN PLACEMENT ENGINE")

    print("=" * 60)

    # --------------------------------------
    # Terrain Settings
    # --------------------------------------

    terrain_size = int(

        input(

            "\nTerrain Grid Size : "

        )

    )

    ground_height = float(

        input(

            "Ground Height : "

        )

    )

    # --------------------------------------
    # Create Objects
    # --------------------------------------

    terrain = TerrainData(

        terrain_size,

        ground_height

    )

    library = PrimitiveLibrary(

        "HEIGHTMAP_FOLDER"

    )

    library.load()

    placement = PlacementEngine(

        terrain,

        library

    )

    # --------------------------------------
    # Start Placement
    # --------------------------------------

    placement.run()

    print("\n========================================")

    print("Placement Completed")

    print("========================================")

    print("\nNow Run:")

    print("python main.py")

    print("\nThen open:")

    print("OUTPUT/thermal_height_grid.npy")

    print("using voxel_viewer.py")