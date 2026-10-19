"""Generate only the additional TN33--TN37 and VN13--VN24 Atlas meshes."""

from pathlib import Path

import gmsh

from FB2_parametric_mesh_generator import (
    WATER_DEPTH,
    generate_case_mesh,
)
from FB2_validation_mesh_generator import validation_fingerprint


TRAINING_RATIOS = (
    (33, 0.50, 1.25),
    (34, 0.50, 1.50),
    (35, 0.50, 1.75),
    (36, 0.60, 1.25),
    (37, 0.60, 1.50),
)

VALIDATION_RATIOS = (
    (13, 0.25, 1.125),
    (14, 0.25, 1.375),
    (15, 0.25, 1.625),
    (16, 0.35, 1.125),
    (17, 0.35, 1.375),
    (18, 0.35, 1.625),
    (19, 0.45, 1.125),
    (20, 0.45, 1.375),
    (21, 0.45, 1.625),
    (22, 0.55, 1.125),
    (23, 0.55, 1.375),
    (24, 0.55, 1.625),
)


def make_training_case(case_number, a_over_h, d_over_a):
    """Create one additional fitting-case definition."""
    a = a_over_h * WATER_DEPTH
    d = d_over_a * a
    return {
        "test_number": case_number,
        "a": a,
        "d": d,
        "h": WATER_DEPTH,
        "a_over_h": a_over_h,
        "d_over_a": d_over_a,
        "d_over_h": d / WATER_DEPTH,
        "atlas_role": "additional Chebyshev fitting case",
    }


def make_validation_case(case_number, a_over_h, d_over_a):
    """Create one additional independent-validation definition."""
    a = a_over_h * WATER_DEPTH
    d = d_over_a * a
    return {
        "case_label": f"VN{case_number:02d}",
        "validation_case_number": case_number,
        "a": a,
        "d": d,
        "h": WATER_DEPTH,
        "a_over_h": a_over_h,
        "d_over_a": d_over_a,
        "d_over_h": d / WATER_DEPTH,
        "validation_role": "independent high-draft equation validation",
        "fingerprint": validation_fingerprint(
            case_number,
            a,
            d,
            WATER_DEPTH,
        ),
    }


def main():
    project_directory = Path(__file__).resolve().parent
    training_directory = project_directory / "FB2_cases"
    validation_directory = project_directory / "FB2_validation_cases"
    training_directory.mkdir(parents=True, exist_ok=True)
    validation_directory.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

        for values in TRAINING_RATIOS:
            generate_case_mesh(
                make_training_case(*values),
                training_directory,
            )

        for values in VALIDATION_RATIOS:
            generate_case_mesh(
                make_validation_case(*values),
                validation_directory,
            )
    finally:
        gmsh.finalize()

    print("TN33--TN37 meshes written to:", training_directory)
    print("VN13--VN24 meshes written to:", validation_directory)


if __name__ == "__main__":
    main()
