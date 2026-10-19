"""Calculate infinite-frequency added mass for the existing FB2 meshes.

This is a separate calculation from ``FB2_parametric_solver.py``.  It does
not evaluate a very small period and it does not repeat diffraction or the
finite-period radiation sweep.  Instead, for each radiation mode it solves
the high-frequency acceleration-potential problem

    Laplace(psi_j) = 0,
    psi_j = 0                         on the free surface,
    d(psi_j)/dn = N_j                on the body,

with homogeneous natural conditions on the remaining boundaries.  The
infinite-frequency added-mass coefficients are then

    A_inf[i,j] = rho * integral_BODY(psi_j * N_i ds).

Each case is saved in a new ``added_mass_infinite`` subdirectory.  Existing
finite-period results are never overwritten.
"""

import argparse
import gc
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh


SURFACE = 9
BODY = 10

GRAVITY = 9.81
WATER_DENSITY = 1000.0
MODE_NUMBERS = (1, 3, 5)
CASE_GROUPS = {
    "TN": "FB2_cases",
    "VN": "FB2_validation_cases",
    "GV": "FB2_Gesraha_validation",
}


def coefficient_scales(a, d, rho):
    """Return the same added-mass scales used by the finite-period solver."""
    translation_mass_scale = 2.0 * rho * a * d
    pitch_inertia_scale = 8.0 * rho * a**3 * d
    mode_scales = np.array((
        translation_mass_scale,
        translation_mass_scale,
        pitch_inertia_scale,
    ))
    return np.sqrt(np.outer(mode_scales, mode_scales))


def assemble_potential_integrals(domain, ds, potential, modes, rho):
    """Integrate one radiation potential against all generalized modes."""
    values = []
    for mode in modes:
        local_value = fem.assemble_scalar(
            fem.form(rho * potential * mode * ds(BODY))
        )
        values.append(domain.comm.allreduce(local_value, op=MPI.SUM))
    return np.asarray(values, dtype=PETSc.ScalarType)


def solve_infinite_radiation(
    domain,
    dx,
    ds,
    V,
    free_surface_bc,
    body_mode,
    output_modes,
    rho,
    petsc_prefix,
):
    """Solve one high-frequency radiation problem and return one A_inf column."""
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L_form = ufl.inner(body_mode, v) * ds(BODY)

    problem = LinearProblem(
        a_form,
        L_form,
        bcs=[free_surface_bc],
        petsc_options_prefix=petsc_prefix,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    potential = problem.solve()

    solver_reason = problem.solver.getConvergedReason()
    if solver_reason <= 0:
        raise RuntimeError(
            "Infinite-frequency radiation solve failed; "
            f"PETSc reason={solver_reason}"
        )

    return assemble_potential_integrals(
        domain,
        ds,
        potential,
        output_modes,
        rho,
    )


def save_case_results(
    case_directory,
    fingerprint,
    a,
    d,
    depth,
    added_mass_infinite,
    added_mass_infinite_nondimensional,
    imaginary_residual,
    elapsed_seconds,
):
    """Write separate NPZ and one-row CSV files for one geometry."""
    output_directory = case_directory / "added_mass_infinite"
    output_directory.mkdir(parents=True, exist_ok=True)

    npz_path = output_directory / f"{fingerprint}_added_mass_infinite.npz"
    csv_path = output_directory / f"{fingerprint}_added_mass_infinite.csv"

    np.savez_compressed(
        npz_path,
        fingerprint=np.array(fingerprint),
        a=np.array(a),
        d=np.array(d),
        h=np.array(depth),
        gravity=np.array(GRAVITY),
        rho=np.array(WATER_DENSITY),
        free_surface_condition=np.array("phi=0"),
        added_mass_infinite=added_mass_infinite,
        added_mass_infinite_nondimensional=(
            added_mass_infinite_nondimensional
        ),
        maximum_imaginary_residual=np.array(imaginary_residual),
        elapsed_seconds=np.array(elapsed_seconds),
        completed_utc=np.array(datetime.now(timezone.utc).isoformat()),
    )

    values = [a, d, depth, a / depth, d / a, d / depth]
    headers = ["a", "d", "h", "a_over_h", "d_over_a", "d_over_h"]
    for matrix_name, matrix in (
        ("Ainf", added_mass_infinite),
        ("Ainf_star", added_mass_infinite_nondimensional),
    ):
        for row, force_mode in enumerate(MODE_NUMBERS):
            for column, motion_mode in enumerate(MODE_NUMBERS):
                headers.append(f"{matrix_name}{force_mode}{motion_mode}")
                values.append(matrix[row, column])
    headers.extend(("maximum_imaginary_residual", "elapsed_seconds"))
    values.extend((imaginary_residual, elapsed_seconds))

    np.savetxt(
        csv_path,
        np.asarray(values, dtype=float)[None, :],
        delimiter=",",
        header=",".join(headers),
        comments="",
    )

    return npz_path, csv_path


def solve_case(case_directory, metadata):
    """Calculate A_inf for one existing mesh without a frequency sweep."""
    fingerprint = str(metadata["fingerprint"])
    a = float(metadata["a"])
    d = float(metadata["d"])
    depth = float(metadata["h"])
    mesh_path = case_directory / metadata["mesh_file"]

    mesh_data = gmsh.read_from_msh(
        str(mesh_path),
        MPI.COMM_WORLD,
        rank=0,
        gdim=2,
    )
    domain = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    n = ufl.FacetNormal(domain)
    x = ufl.SpatialCoordinate(domain)

    x_ref = 0.0
    z_ref = 0.0
    modes = (
        n[0],
        n[1],
        (x[1] - z_ref) * n[0] - (x[0] - x_ref) * n[1],
    )

    V = fem.functionspace(domain, ("Lagrange", 1))
    facet_dimension = domain.topology.dim - 1
    surface_facets = facet_tags.find(SURFACE)
    if surface_facets.size == 0:
        raise RuntimeError(
            f"{fingerprint}: no facets have the free-surface tag {SURFACE}."
        )
    surface_dofs = fem.locate_dofs_topological(
        V,
        facet_dimension,
        surface_facets,
    )
    zero_potential = fem.Function(V)
    zero_potential.x.array[:] = PETSc.ScalarType(0.0)
    free_surface_bc = fem.dirichletbc(zero_potential, surface_dofs)

    start_time = MPI.Wtime()
    complex_matrix = np.zeros((3, 3), dtype=PETSc.ScalarType)
    for column, (mode_number, body_mode) in enumerate(
        zip(MODE_NUMBERS, modes)
    ):
        complex_matrix[:, column] = solve_infinite_radiation(
            domain,
            dx,
            ds,
            V,
            free_surface_bc,
            body_mode,
            modes,
            WATER_DENSITY,
            f"{fingerprint}_Ainf_radiation_{mode_number}_",
        )

    domain.comm.Barrier()
    elapsed_seconds = MPI.Wtime() - start_time
    imaginary_residual = float(np.max(np.abs(complex_matrix.imag)))
    added_mass_infinite = complex_matrix.real
    scales = coefficient_scales(a, d, WATER_DENSITY)
    added_mass_infinite_nondimensional = added_mass_infinite / scales

    if domain.comm.rank == 0:
        npz_path, _ = save_case_results(
            case_directory,
            fingerprint,
            a,
            d,
            depth,
            added_mass_infinite,
            added_mass_infinite_nondimensional,
            imaginary_residual,
            elapsed_seconds,
        )
        print(
            f"{fingerprint}: A_inf completed in {elapsed_seconds:.2f} s; "
            f"saved to {npz_path}"
        )

    del domain
    gc.collect()


def all_case_directories(project_directory, group="all"):
    """Return existing TN, VN and GV geometry directories."""
    selected_groups = CASE_GROUPS if group == "all" else {group: CASE_GROUPS[group]}
    directories = []
    for prefix, relative_directory in selected_groups.items():
        root = project_directory / relative_directory
        directories.extend(
            path
            for path in root.glob(f"{prefix}[0-9][0-9]_*")
            if path.is_dir()
        )
    return sorted(directories, key=lambda path: path.name)


def select_case(project_directory, requested_case):
    """Find one TN, VN or GV case from a short name such as TN10."""
    name = requested_case.upper()
    if name.isdigit():
        name = f"TN{int(name):02d}"
    elif len(name) >= 3 and name[:2] in CASE_GROUPS:
        name = f"{name[:2]}{int(name[2:]):02d}"
    else:
        raise ValueError(
            "--case must be a number or a TN, VN or GV name, for example TN10."
        )

    matches = [
        path
        for path in all_case_directories(project_directory)
        if path.name.startswith(f"{name}_")
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one folder for {name}; found {len(matches)}.")
    return matches


def result_path(case_directory, metadata):
    fingerprint = str(metadata["fingerprint"])
    return (
        case_directory
        / "added_mass_infinite"
        / f"{fingerprint}_added_mass_infinite.npz"
    )


def solve_selected_case(case_directory, overwrite):
    metadata_path = case_directory / "case_parameters.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output_path = result_path(case_directory, metadata)
    if output_path.exists() and not overwrite:
        if MPI.COMM_WORLD.rank == 0:
            print(f"{metadata['fingerprint']}: existing A_inf result found; skipped")
        return
    solve_case(case_directory, metadata)


def run_parallel_cases(project_directory, case_directories, workers, overwrite):
    """Run independent cases in separate processes and FEniCS caches."""
    case_names = [path.name.split("_", 1)[0] for path in case_directories]
    worker_groups = [case_names[index::workers] for index in range(workers)]
    solver_path = Path(__file__).resolve()
    cache_root = project_directory / ".fenics_Ainf_worker_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    def run_worker(worker_number, names):
        environment = os.environ.copy()
        environment["XDG_CACHE_HOME"] = str(
            cache_root / f"worker_{worker_number + 1}"
        )
        for name in names:
            command = [sys.executable, str(solver_path), "--case", name]
            if overwrite:
                command.append("--overwrite")
            completed = subprocess.run(
                command,
                cwd=project_directory,
                env=environment,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Worker {worker_number + 1}: {name} failed with "
                    f"return code {completed.returncode}."
                )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_worker, index, names)
            for index, names in enumerate(worker_groups)
            if names
        ]
        for future in futures:
            future.result()


def write_project_summary(project_directory):
    """Collect every available per-case A_inf result into one CSV table."""
    rows = []
    for case_directory in all_case_directories(project_directory):
        metadata = json.loads(
            (case_directory / "case_parameters.json").read_text(encoding="utf-8")
        )
        path = result_path(case_directory, metadata)
        if not path.exists():
            continue
        result = np.load(path)
        matrix = result["added_mass_infinite"]
        matrix_star = result["added_mass_infinite_nondimensional"]
        row = [
            str(result["fingerprint"]),
            float(result["a"]),
            float(result["d"]),
            float(result["h"]),
            float(result["a"] / result["h"]),
            float(result["d"] / result["a"]),
        ]
        row.extend(matrix.ravel().tolist())
        row.extend(matrix_star.ravel().tolist())
        rows.append(row)

    if not rows:
        return

    headers = ["case", "a", "d", "h", "a_over_h", "d_over_a"]
    for prefix in ("Ainf", "Ainf_star"):
        for force_mode in MODE_NUMBERS:
            for motion_mode in MODE_NUMBERS:
                headers.append(f"{prefix}{force_mode}{motion_mode}")

    summary_path = project_directory / "FB2_added_mass_infinite_summary.csv"
    with summary_path.open("w", encoding="utf-8") as stream:
        stream.write(",".join(headers) + "\n")
        for row in rows:
            stream.write(",".join(str(value) for value in row) + "\n")
    print(f"Combined summary written to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate infinite-frequency added mass for existing FB2 meshes "
            "without repeating the finite-period calculations."
        )
    )
    parser.add_argument(
        "--case",
        help="Optional single case, for example TN10, VN08, GV01, or 10.",
    )
    parser.add_argument(
        "--group",
        choices=("all", "TN", "VN", "GV"),
        default="all",
        help="Geometry group used when --case is omitted; default is all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of independent case workers; default is 4.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recalculate an existing infinite-added-mass result.",
    )
    arguments = parser.parse_args()

    if arguments.workers < 1:
        raise ValueError("--workers must be at least 1.")

    project_directory = Path(__file__).resolve().parent
    if arguments.case:
        directories = select_case(project_directory, arguments.case)
        for directory in directories:
            solve_selected_case(directory, arguments.overwrite)
    else:
        if MPI.COMM_WORLD.size != 1:
            raise RuntimeError(
                "Run the automatic multi-case launcher with ordinary Python, "
                "not mpirun."
            )
        directories = all_case_directories(project_directory, arguments.group)
        run_parallel_cases(
            project_directory,
            directories,
            min(arguments.workers, len(directories)),
            arguments.overwrite,
        )

    if MPI.COMM_WORLD.rank == 0:
        write_project_summary(project_directory)
        print("Requested infinite-frequency added-mass cases are complete.")


if __name__ == "__main__":
    main()
