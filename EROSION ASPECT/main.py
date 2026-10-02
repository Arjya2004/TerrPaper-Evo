# ==========================================================
# TERRAIN CA - ONE COMMAND PIPELINE
# ==========================================================
#
# RUN:
#
#     python placement_engine.py
#
# তারপর:
#
#     python main.py
#
# User শুধু ITERATIONS দেবে।
# অন্য সব parameter automatically দেওয়া হবে।
#
# ==========================================================

import builtins
import runpy
from pathlib import Path

from config import (
    ITERATIONS,

    # R1
    BLENDING_STRENGTH,

    # R2
    TALUS_THRESHOLD,
    EROSION_STRENGTH,
    MAXIMUM_MATERIAL_MOVE,

    # R3
    DEPOSITION_RATE,
    TRANSPORT_THRESHOLD,
    DEPOSITION_THRESHOLD,

    # Hydraulic
    RAIN_AMOUNT,
    EVAPORATION_RATE,
    SEDIMENT_CAPACITY,
    HYDRAULIC_EROSION_RATE,
    HYDRAULIC_DEPOSITION_RATE,

    # R5
    SLOPE_THRESHOLD,
    STEEP_THRESHOLD,
    PLATEAU_THRESHOLD,
    PEAK_THRESHOLD,
    SEDIMENT_THRESHOLD
)


# ==========================================================
# PROJECT PATH
# ==========================================================

ROOT = Path(
    __file__
).resolve().parent

OUTPUT = (
    ROOT /
    "OUTPUT"
)


# ==========================================================
# CURRENT ITERATION VALUE
# ==========================================================

RUN_ITERATIONS = ITERATIONS


# ==========================================================
# AUTOMATIC INPUT HANDLER
# ==========================================================

def auto_answer(prompt):

    p = str(
        prompt
    ).lower().strip()


    # ======================================================
    # ITERATIONS
    # ======================================================

    if "transport iterations" in p:

        return str(
            RUN_ITERATIONS
        )


    if "sediment iterations" in p:

        return str(
            RUN_ITERATIONS
        )


    if "hydraulic iterations" in p:

        return str(
            RUN_ITERATIONS
        )


    if "rainfall iterations" in p:

        return str(
            RUN_ITERATIONS
        )


    if "erosion iterations" in p:

        return str(
            RUN_ITERATIONS
        )


    if "thermal iterations" in p:

        return str(
            RUN_ITERATIONS
        )


    if "blending iterations" in p:

        return str(
            RUN_ITERATIONS
        )


    if "final ca iterations" in p:

        return str(
            RUN_ITERATIONS
        )


    if "final iterations" in p:

        return str(
            RUN_ITERATIONS
        )


    if (
        p.endswith("iterations :")
        or
        p.endswith("iterations:")
    ):

        return str(
            RUN_ITERATIONS
        )


    # ======================================================
    # R1 - BLENDING
    # ======================================================

    if "blending strength" in p:

        return str(
            BLENDING_STRENGTH
        )


    if "transition strength" in p:

        return str(
            BLENDING_STRENGTH
        )


    # ======================================================
    # R2 - THERMAL / EROSION
    # ======================================================

    if "talus angle" in p:

        return str(
            TALUS_THRESHOLD
        )


    if "talus threshold" in p:

        return str(
            TALUS_THRESHOLD
        )


    if "thermal strength" in p:

        return str(
            EROSION_STRENGTH
        )


    if "erosion strength" in p:

        return str(
            EROSION_STRENGTH
        )


    if "maximum material move" in p:

        return str(
            MAXIMUM_MATERIAL_MOVE
        )


    if "maximum move" in p:

        return str(
            MAXIMUM_MATERIAL_MOVE
        )


    if "max move" in p:

        return str(
            MAXIMUM_MATERIAL_MOVE
        )


       # ======================================================
    # R3 - SEDIMENT
    # ======================================================

    if "deposition rate" in p:

        return str(
            DEPOSITION_RATE
        )


    if "transport threshold" in p:

        return str(
            TRANSPORT_THRESHOLD
        )


    if "transport rate" in p:

        return str(
            TRANSPORT_THRESHOLD
        )


    if "deposition threshold" in p:

        return str(
            DEPOSITION_THRESHOLD
        )


    if "maximum deposit" in p:

        return str(
            DEPOSITION_THRESHOLD
        )
       # ======================================================
    # R4 - HYDRAULIC EROSION
    # ======================================================

    if "hydraulic iterations" in p:

        return str(
            RUN_ITERATIONS
        )


    if "rain amount" in p:

        return str(
            RAIN_AMOUNT
        )


    if "rainfall amount" in p:

        return str(
            RAIN_AMOUNT
        )


    if "rain frequency" in p:

        return "1"


    if "flow rate" in p:

        return "0.20"


    if "evaporation rate" in p:

        return str(
            EVAPORATION_RATE
        )


    if "evaporation" in p:

        return str(
            EVAPORATION_RATE
        )


    if "soil softness" in p:

        return "0.20"


    if "sediment capacity" in p:

        return str(
            SEDIMENT_CAPACITY
        )


    if "carrying capacity" in p:

        return str(
            SEDIMENT_CAPACITY
        )


    if "hydraulic erosion rate" in p:

        return str(
            HYDRAULIC_EROSION_RATE
        )


    if "maximum hydraulic erosion" in p:

        return "0.50"


    if "hydraulic deposition rate" in p:

        return str(
            HYDRAULIC_DEPOSITION_RATE
        )


    if "maximum hydraulic deposition" in p:

        return "0.50"
            # ======================================================
    # R5 - FINAL TERRAIN STATE INTEGRATION
    # ======================================================

    if "final ca iterations" in p:

        return str(
            RUN_ITERATIONS
        )


    if "final iterations" in p:

        return str(
            RUN_ITERATIONS
        )


    if "slope threshold" in p:

        return str(
            SLOPE_THRESHOLD
        )


    if "steep threshold" in p:

        return str(
            STEEP_THRESHOLD
        )


    if "plateau threshold" in p:

        return str(
            PLATEAU_THRESHOLD
        )


    if "peak threshold" in p:

        return str(
            PEAK_THRESHOLD
        )


    if "sediment threshold" in p:

        return str(
            SEDIMENT_THRESHOLD
        )
    # ======================================================
    # UNKNOWN PROMPT
    # ======================================================

    raise RuntimeError(

        "\nUnknown parameter prompt detected:\n"

        f"    {prompt!r}\n\n"

        "Add this prompt to auto_answer() "
        "before running the pipeline again."
    )


# ==========================================================
# RUN ONE PIPELINE STAGE
# ==========================================================

def run_stage(
    script_name
):

    script_path = (
        ROOT /
        script_name
    )


    if not script_path.exists():

        raise FileNotFoundError(

            f"\nRequired file not found:\n"
            f"{script_name}"
        )


    print(
        "\n" + "=" * 70
    )

    print(
        f"RUNNING: {script_name}"
    )

    print(
        "=" * 70
    )


    # ------------------------------------------------------
    # Save original input()
    # ------------------------------------------------------

    original_input = (
        builtins.input
    )


    # ------------------------------------------------------
    # Automatic input()
    # ------------------------------------------------------

    def automatic_input(
        prompt=""
    ):

        answer = auto_answer(
            prompt
        )


        print(
            f"{prompt}"
            f"[AUTO: {answer}]"
        )


        return answer


    # ------------------------------------------------------
    # Replace input()
    # ------------------------------------------------------

    builtins.input = (
        automatic_input
    )


    try:

        runpy.run_path(

            str(
                script_path
            ),

            run_name="__main__"
        )


    finally:

        # Always restore original input()
        builtins.input = (
            original_input
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    global RUN_ITERATIONS


    print(
        "\n" + "=" * 70
    )

    print(
        "TERRAIN CA - ONE COMMAND PIPELINE"
    )

    print(
        "=" * 70
    )


    # ======================================================
    # ONLY USER INPUT
    # ======================================================

    while True:

        raw = input(

            f"\nIterations "
            f"[{ITERATIONS}]: "

        ).strip()


        # --------------------------------------------------
        # Press Enter = default
        # --------------------------------------------------

        if raw == "":

            RUN_ITERATIONS = (
                ITERATIONS
            )

            break


        # --------------------------------------------------
        # Convert to integer
        # --------------------------------------------------

        try:

            value = int(
                raw
            )

        except ValueError:

            print(
                "Enter a whole number."
            )

            continue


        # --------------------------------------------------
        # Validate
        # --------------------------------------------------

        if value <= 0:

            print(
                "Iterations must be greater than zero."
            )

            continue


        RUN_ITERATIONS = (
            value
        )

        break


    print(
        "\nSelected Iterations :",
        RUN_ITERATIONS
    )


    # ======================================================
    # CHECK PLACEMENT OUTPUT
    # ======================================================

    height_file = (
        OUTPUT /
        "height_grid.npy"
    )


    if not height_file.exists():

        print(
            "\nERROR:"
        )

        print(
            "OUTPUT/height_grid.npy "
            "was not found."
        )

        print(
            "\nFirst run:"
        )

        print(
            "python placement_engine.py"
        )

        return 1


    # ======================================================
    # PIPELINE
    # ======================================================

    pipeline = [

        "ca_state.py",

        "ca_rules.py",

        "ca_blending.py",

        "ca_erosion.py",

        "ca_sediment.py",

        "ca_hydraulic.py",

        "ca_final.py"

    ]


    # ======================================================
    # RUN ALL STAGES
    # ======================================================

    try:

        for script in pipeline:

            run_stage(
                script
            )


    except Exception as error:

        print(
            "\n" + "=" * 70
        )

        print(
            "PIPELINE STOPPED"
        )

        print(
            "=" * 70
        )

        print(
            "\nError:"
        )

        print(
            error
        )

        return 1


    # ======================================================
    # FINAL OUTPUT
    # ======================================================

    final_height = (
        OUTPUT /
        "final_height_grid.npy"
    )

    final_state = (
        OUTPUT /
        "final_state_grid.npy"
    )

    final_sediment = (
        OUTPUT /
        "final_sediment_grid.npy"
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "TERRAIN CA PIPELINE COMPLETED"
    )

    print(
        "=" * 70
    )


    print(
        "\nFinal Height:"
    )

    print(
        final_height
    )


    print(
        "\nFinal State:"
    )

    print(
        final_state
    )


    print(
        "\nFinal Sediment:"
    )

    print(
        final_sediment
    )


    print(
        "\nAll CA stages completed successfully."
    )


    return 0


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )