# TerrCA-Evo

After ALL that CHAOS trying to keep things stable with my two brain cells.

## Overview

A Cellular Automata-based framework for procedural three-dimensional
terrain generation and terrain evolution.

The current workflow explores:

- One-dimensional Cellular Automata for vertical profile generation
- Three-dimensional rotational terrain construction
- Voronoi-based spatial variation
- Terrain placement
- State-aware terrain processing
- Thermal/gradient erosion
- Sediment transport and deposition
- Hydraulic erosion
- Final terrain evolution

## Pipeline

```text
1D Cellular Automaton
        ↓
vertical_profile.npy
        ↓
Rotational Construction
        ↓
rotation_heightmap.npy
        ↓
Voronoi Variation
        ↓
variation_heightmap.npy
        ↓
Terrain Placement
        ↓
Boundary Blending
        ↓
Gradient Erosion
        ↓
Sediment Transport
        ↓
Hydraulic Erosion
        ↓
Final Terrain