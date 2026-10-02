# ==========================================================
# TERRAIN CA - CENTRAL CONFIGURATION
# ==========================================================
# Normally you only change ITERATIONS.
# All other values are fixed defaults for the current pipeline.

ITERATIONS = 30

# R1 - blending
BLENDING_STRENGTH = 0.20

# R2 - state-aware gradient / thermal erosion
TALUS_THRESHOLD = 1.0
EROSION_STRENGTH = 0.20
MAXIMUM_MATERIAL_MOVE = 0.50

# R3 - sediment
DEPOSITION_RATE = 0.35
TRANSPORT_THRESHOLD = 0.20
DEPOSITION_THRESHOLD = 0.50

# Hydraulic erosion
RAIN_AMOUNT = 0.50
EVAPORATION_RATE = 0.02
SEDIMENT_CAPACITY = 0.50
HYDRAULIC_EROSION_RATE = 0.10
HYDRAULIC_DEPOSITION_RATE = 0.10

# R5 - final state integration
SLOPE_THRESHOLD = 0.45
STEEP_THRESHOLD = 0.70
PLATEAU_THRESHOLD = 0.20
PEAK_THRESHOLD = 0.80
SEDIMENT_THRESHOLD = 0.05