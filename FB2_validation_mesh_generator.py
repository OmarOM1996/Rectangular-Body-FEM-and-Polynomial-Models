"""Generate the 12 meshes used for independent equation validation."""

from pathlib import Path

import gmsh

from FB2_parametric_mesh_generator import (
    WATER_DEPTH,
    generate_case_mesh,
    number_for_name,
)


A_OVER_H_VALUES = (0.25, 0.35, 0.45, 0.55)
D_OVER_A_VALUES = (0.375, 0.625, 0.875)


def validation_fingerprint(case_number, a, d, h):
    """Create the common folder and file prefix for one VN case."""
    return (
        f"VN{case_number:02d}"
        f"_a{number_for_name(a)}"
        f"_d{number_for_name(d)}"
        f"_h{number_for_name(h)}"
        f"_ah{number_for_name(a / h)}"
        f"_da{number_for_name(d / a)}"
    )


def validation_cases():
    """Return the 12 geometries located between the original TN cases."""
    cases = []
    case_number = 1

    for a_over_h in A_OVER_H_VALUES:
        a = a_over_h * WATER_DEPTH

        for d_over_a in D_OVER_A_VALUES:
            d = d_over_a * a
            cases.append({
                "case_label": f"VN{case_number:02d}",
                "validation_case_number": case_number,
                "a": a,
                "d": d,
                "h": WATER_DEPTH,
                "a_over_h": a_over_h,
                "d_over_a": d_over_a,
                "d_over_h": d / WATER_DEPTH,
                "validation_role": "independent equation validation",
                "fingerprint": validation_fingerprint(
                    case_number,
                    a,
                    d,
                    WATER_DEPTH,
                ),
            })
            case_number += 1

    return cases


def main():
    project_directory = Path(__file__).resolve().parent
    validation_directory = project_directory / "FB2_validation_cases"
    validation_directory.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

        for case in validation_cases():
            generate_case_mesh(case, validation_directory)
    finally:
        gmsh.finalize()

    print("Twelve validation meshes written to:", validation_directory)


if __name__ == "__main__":
    main()
