import json
from pathlib import Path

import gmsh


SURFACE = 9
BODY = 10
BOTTOM = 11
LEFT = 12
RIGHT = 13
LOWER = 14

WATER_DEPTH = 5.0
HALF_DOMAIN_WIDTH = 20.0
OUTER_MESH_SIZE = 0.1

A_OVER_H_VALUES = (0.20, 0.30, 0.40, 0.50, 0.60)
D_OVER_A_VALUES = (0.25, 0.50, 0.75, 1.00)

# Deep-draft additions requested after the original 20-case study.
EXTENDED_A_OVER_H_VALUES = (0.20, 0.30, 0.40)
EXTENDED_D_OVER_A_VALUES = (1.25, 1.50, 1.75, 2.00)


def number_for_name(value):
    """Format a number with two decimals and replace the decimal point by p."""
    return f"{value:.2f}".replace(".", "p")


def case_fingerprint(test_number, a, d, h):
    """Create the common folder and file prefix for one case."""
    return (
        f"TN{test_number:02d}"
        f"_a{number_for_name(a)}"
        f"_d{number_for_name(d)}"
        f"_h{number_for_name(h)}"
        f"_ah{number_for_name(a / h)}"
        f"_da{number_for_name(d / a)}"
    )


def validation_fingerprint(a, d, h):
    """Create the folder and file prefix for the Gesraha validation case."""
    return (
        "GV01"
        f"_a{number_for_name(a)}"
        f"_d{number_for_name(d)}"
        f"_h{number_for_name(h)}"
        f"_ah{number_for_name(a / h)}"
        f"_da{number_for_name(d / a)}"
    )


def predefined_cases():
    """Return the original 20 cases followed by the 12 deep-draft additions."""
    cases = []
    test_number = 1

    for a_over_h in A_OVER_H_VALUES:
        a = a_over_h * WATER_DEPTH

        for d_over_a in D_OVER_A_VALUES:
            d = d_over_a * a
            cases.append({
                "test_number": test_number,
                "a": a,
                "d": d,
                "h": WATER_DEPTH,
                "a_over_h": a_over_h,
                "d_over_a": d_over_a,
                "d_over_h": d / WATER_DEPTH,
            })
            test_number += 1

    for a_over_h in EXTENDED_A_OVER_H_VALUES:
        a = a_over_h * WATER_DEPTH

        for d_over_a in EXTENDED_D_OVER_A_VALUES:
            d = d_over_a * a
            cases.append({
                "test_number": test_number,
                "a": a,
                "d": d,
                "h": WATER_DEPTH,
                "a_over_h": a_over_h,
                "d_over_a": d_over_a,
                "d_over_h": d / WATER_DEPTH,
            })
            test_number += 1

    return cases


def gesraha_validation_case():
    """Return the Figure 3 geometry: a/h = 0.10 and d/h = 0.10."""
    a = 0.10 * WATER_DEPTH
    d = 0.10 * WATER_DEPTH
    return {
        "case_label": "GV01",
        "a": a,
        "d": d,
        "h": WATER_DEPTH,
        "a_over_h": a / WATER_DEPTH,
        "d_over_a": d / a,
        "d_over_h": d / WATER_DEPTH,
        "source": "Gesraha Figure 3 validation geometry",
        "fingerprint": validation_fingerprint(a, d, WATER_DEPTH),
    }


def generate_case_mesh(case, cases_directory):
    """Generate the tagged Gmsh mesh and metadata for one predefined case."""
    if "case_label" in case:
        case_label = case["case_label"]
    else:
        case_label = f"TN{case['test_number']:02d}"
    a = case["a"]
    d = case["d"]
    h = case["h"]

    if not 0.0 < a < HALF_DOMAIN_WIDTH:
        raise ValueError(f"{case_label}: a must satisfy 0 < a < 20.")

    if not 0.0 < d < h:
        raise ValueError(f"{case_label}: d must satisfy 0 < d < h.")

    if "fingerprint" in case:
        fingerprint = case["fingerprint"]
    else:
        fingerprint = case_fingerprint(case["test_number"], a, d, h)
    case_directory = cases_directory / fingerprint
    case_directory.mkdir(parents=True, exist_ok=True)

    mesh_path = case_directory / f"{fingerprint}_mesh.msh"
    metadata_path = case_directory / "case_parameters.json"

    body_mesh_size = min(
        OUTER_MESH_SIZE,
        a / 20.0,
        d / 10.0,
    )

    gmsh.clear()
    gmsh.model.add(fingerprint)

    point_left_surface = gmsh.model.geo.addPoint(
        -HALF_DOMAIN_WIDTH, 0.0, 0.0, OUTER_MESH_SIZE
    )
    point_body_left_top = gmsh.model.geo.addPoint(
        -a, 0.0, 0.0, body_mesh_size
    )
    point_body_left_bottom = gmsh.model.geo.addPoint(
        -a, -d, 0.0, body_mesh_size
    )
    point_body_right_bottom = gmsh.model.geo.addPoint(
        a, -d, 0.0, body_mesh_size
    )
    point_body_right_top = gmsh.model.geo.addPoint(
        a, 0.0, 0.0, body_mesh_size
    )
    point_right_surface = gmsh.model.geo.addPoint(
        HALF_DOMAIN_WIDTH, 0.0, 0.0, OUTER_MESH_SIZE
    )
    point_right_bottom = gmsh.model.geo.addPoint(
        HALF_DOMAIN_WIDTH, -h, 0.0, OUTER_MESH_SIZE
    )
    point_left_bottom = gmsh.model.geo.addPoint(
        -HALF_DOMAIN_WIDTH, -h, 0.0, OUTER_MESH_SIZE
    )

    surface_left = gmsh.model.geo.addLine(
        point_left_surface, point_body_left_top
    )
    body_left = gmsh.model.geo.addLine(
        point_body_left_top, point_body_left_bottom
    )
    body_bottom = gmsh.model.geo.addLine(
        point_body_left_bottom, point_body_right_bottom
    )
    body_right = gmsh.model.geo.addLine(
        point_body_right_bottom, point_body_right_top
    )
    surface_right = gmsh.model.geo.addLine(
        point_body_right_top, point_right_surface
    )
    right_boundary = gmsh.model.geo.addLine(
        point_right_surface, point_right_bottom
    )
    bottom_boundary = gmsh.model.geo.addLine(
        point_right_bottom, point_left_bottom
    )
    left_boundary = gmsh.model.geo.addLine(
        point_left_bottom, point_left_surface
    )

    boundary_loop = gmsh.model.geo.addCurveLoop((
        surface_left,
        body_left,
        body_bottom,
        body_right,
        surface_right,
        right_boundary,
        bottom_boundary,
        left_boundary,
    ))
    fluid_domain = gmsh.model.geo.addPlaneSurface((boundary_loop,))

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, (surface_left, surface_right), SURFACE)
    gmsh.model.setPhysicalName(1, SURFACE, "SURFACE")

    gmsh.model.addPhysicalGroup(
        1, (body_left, body_bottom, body_right), BODY
    )
    gmsh.model.setPhysicalName(1, BODY, "BODY")

    gmsh.model.addPhysicalGroup(1, (bottom_boundary,), BOTTOM)
    gmsh.model.setPhysicalName(1, BOTTOM, "BOTTOM")

    gmsh.model.addPhysicalGroup(1, (left_boundary,), LEFT)
    gmsh.model.setPhysicalName(1, LEFT, "LEFT")

    gmsh.model.addPhysicalGroup(1, (right_boundary,), RIGHT)
    gmsh.model.setPhysicalName(1, RIGHT, "RIGHT")

    gmsh.model.addPhysicalGroup(2, (fluid_domain,), LOWER)
    gmsh.model.setPhysicalName(2, LOWER, "LOWER")

    gmsh.model.mesh.generate(2)
    gmsh.write(str(mesh_path))

    metadata = {
        **case,
        "fingerprint": fingerprint,
        "half_domain_width": HALF_DOMAIN_WIDTH,
        "outer_mesh_size": OUTER_MESH_SIZE,
        "body_mesh_size": body_mesh_size,
        "mesh_file": mesh_path.name,
        "physical_tags": {
            "SURFACE": SURFACE,
            "BODY": BODY,
            "BOTTOM": BOTTOM,
            "LEFT": LEFT,
            "RIGHT": RIGHT,
            "LOWER": LOWER,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    node_tags, _, _ = gmsh.model.mesh.getNodes()
    print(
        f"{fingerprint}: {len(node_tags)} nodes, "
        f"body mesh size {body_mesh_size:.4f} m"
    )


def main():
    project_directory = Path(__file__).resolve().parent
    cases_directory = project_directory / "FB2_cases"
    validation_directory = project_directory / "FB2_Gesraha_validation"
    cases_directory.mkdir(parents=True, exist_ok=True)
    validation_directory.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

        for case in predefined_cases():
            generate_case_mesh(case, cases_directory)
        generate_case_mesh(
            gesraha_validation_case(),
            validation_directory,
        )
    finally:
        gmsh.finalize()

    print("Thirty-two case meshes written to:", cases_directory)
    print("Gesraha validation mesh written to:", validation_directory)


if __name__ == "__main__":
    main()
