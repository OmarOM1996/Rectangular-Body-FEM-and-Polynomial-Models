"""Solve VN01--VN12 independently using four parallel case workers."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mpi4py import MPI

from FB2_parametric_solver import TAU_POINTS, solve_case


def selected_validation_directories(project_directory, requested_case):
    """Return all VN folders or one requested VN folder."""
    validation_directory = project_directory / "FB2_validation_cases"
    case_directories = sorted(
        path
        for path in validation_directory.glob("VN*_d*_da*")
        if path.is_dir()
    )

    if requested_case is None:
        return case_directories

    requested_case = requested_case.upper()
    if requested_case.isdigit():
        requested_case = f"VN{int(requested_case):02d}"
    elif requested_case.startswith("VN"):
        requested_case = f"VN{int(requested_case[2:]):02d}"

    matches = [
        path
        for path in case_directories
        if path.name.startswith(f"{requested_case}_")
    ]
    if not matches:
        raise ValueError(f"No validation folder found for {requested_case}.")

    return matches


def run_parallel_validation_cases(project_directory, points, overwrite):
    """Distribute the 12 VN cases over four independent workers."""
    case_names = [f"VN{case_number:02d}" for case_number in range(1, 13)]
    worker_case_groups = [
        case_names[worker_number::4]
        for worker_number in range(4)
    ]
    solver_path = Path(__file__).resolve()
    cache_root = project_directory / ".fenics_validation_worker_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    def run_worker(worker_number, worker_cases):
        worker_environment = os.environ.copy()
        worker_environment["XDG_CACHE_HOME"] = str(
            cache_root / f"worker_{worker_number + 1}"
        )

        for case_name in worker_cases:
            command = [
                sys.executable,
                str(solver_path),
                "--case",
                case_name,
                "--points",
                str(points),
            ]
            if overwrite:
                command.append("--overwrite")

            completed_process = subprocess.run(
                command,
                cwd=project_directory,
                env=worker_environment,
                check=False,
            )
            if completed_process.returncode != 0:
                raise RuntimeError(
                    f"Worker {worker_number + 1}: {case_name} failed "
                    f"with return code {completed_process.returncode}."
                )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_worker, worker_number, worker_cases)
            for worker_number, worker_cases in enumerate(worker_case_groups)
        ]
        for future in futures:
            future.result()


def main():
    parser = argparse.ArgumentParser(
        description="Solve the 12 independent FB2 validation cases."
    )
    parser.add_argument(
        "--case",
        help="Optional single validation case, for example VN05 or 5.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recalculate validation cases whose NPZ result exists.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=TAU_POINTS,
        help="Number of T/Tv points; default is 100.",
    )
    arguments = parser.parse_args()

    if arguments.points < 2:
        raise ValueError("--points must be at least 2.")

    project_directory = Path(__file__).resolve().parent

    if arguments.case is None:
        if MPI.COMM_WORLD.size != 1:
            raise RuntimeError(
                "Run the automatic four-case launcher with ordinary Python, "
                "not mpirun."
            )
        run_parallel_validation_cases(
            project_directory,
            arguments.points,
            arguments.overwrite,
        )
        if MPI.COMM_WORLD.rank == 0:
            print("All 12 independent validation cases are complete.")
        return

    case_directories = selected_validation_directories(
        project_directory,
        arguments.case,
    )

    for case_directory in case_directories:
        metadata_path = case_directory / "case_parameters.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fingerprint = metadata["fingerprint"]
        result_path = case_directory / f"{fingerprint}_results.npz"

        if result_path.exists() and not arguments.overwrite:
            if MPI.COMM_WORLD.rank == 0:
                print(f"{fingerprint}: existing result found; skipped")
            continue

        solve_case(case_directory, metadata, arguments.points)


if __name__ == "__main__":
    main()
