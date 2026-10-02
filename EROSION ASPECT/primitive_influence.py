import numpy as np
from collections import deque

class PrimitiveInfluence:

    def __init__(

        self,
        height_grid,
        fixed_mask,
        terrain_type

    ):

        self.height = height_grid

        self.fixed = fixed_mask

        self.type = terrain_type

        self.rows, self.cols = self.height.shape

        self.distance_grid = np.full(

            (self.rows, self.cols),

            np.inf,

            dtype=float

        )

        self.influence_grid = np.zeros(

            (self.rows, self.cols),

            dtype=float

        )

        self.nearest_type_grid = np.zeros(

            (self.rows, self.cols),

            dtype=np.uint8

        )

        self.nearest_height_grid = np.zeros(

            (self.rows, self.cols),

            dtype=float

        )

    # ----------------------------------------

    def primitive_cells(self):

        cells = []

        for r in range(self.rows):

            for c in range(self.cols):

                if self.fixed[r, c]:

                    cells.append(

                        (

                            r,
                            c,

                            self.type[r, c],

                            self.height[r, c]

                        )

                    )

        return cells
    from collections import deque

    # --------------------------------------------------
    # Multi-Source BFS
    # --------------------------------------------------

    def generate(self):

        queue = deque()

        visited = np.zeros(

            (self.rows, self.cols),

            dtype=bool

        )

        directions = [

            (-1,0),

            (1,0),

            (0,-1),

            (0,1),

            (-1,-1),

            (-1,1),

            (1,-1),

            (1,1)

        ]

        # -----------------------------
        # Push all primitive cells
        # -----------------------------

        for r, c, t, h in self.primitive_cells():

            queue.append((r, c))

            visited[r, c] = True

            self.distance_grid[r, c] = 0

            self.nearest_type_grid[r, c] = t

            self.nearest_height_grid[r, c] = h

        # -----------------------------
        # BFS Expansion
        # -----------------------------

        while queue:

            r, c = queue.popleft()

            for dr, dc in directions:

                nr = r + dr
                nc = c + dc

                if nr < 0 or nr >= self.rows:
                    continue

                if nc < 0 or nc >= self.cols:
                    continue

                if visited[nr, nc]:
                    continue

                visited[nr, nc] = True

                self.distance_grid[nr, nc] = (

                    self.distance_grid[r, c] + 1

                )

                self.nearest_type_grid[nr, nc] = (

                    self.nearest_type_grid[r, c]

                )

                self.nearest_height_grid[nr, nc] = (

                    self.nearest_height_grid[r, c]

                )

                queue.append((nr, nc))
                    # --------------------------------------------------
    # Gaussian Influence Map
    # --------------------------------------------------

    def build_influence_map(self, sigma=15.0):

        sigma2 = 2.0 * sigma * sigma

        self.influence_grid = np.exp(

            -(self.distance_grid ** 2) / sigma2

        )

        self.influence_grid[self.fixed] = 1.0

        return self.influence_grid

    # --------------------------------------------------
    # Get Results
    # --------------------------------------------------

    def get_results(self):

        return (

            self.distance_grid,

            self.influence_grid,

            self.nearest_type_grid,

            self.nearest_height_grid

        )

    # --------------------------------------------------
    # Run
    # --------------------------------------------------

    def run(self, sigma=15.0):

        self.generate()

        self.build_influence_map(sigma)

        return self.get_results()