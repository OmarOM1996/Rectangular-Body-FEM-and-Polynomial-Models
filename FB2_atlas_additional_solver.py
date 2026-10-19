"""Solve TN33--TN37 and VN13--VN24 with four independent case workers."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ADDITIONAL_CASES = (
    *(f"TN{case_number:02d}" for case_number in range(33, 38)),
    *(f"VN{case_number:02d}" for case_number in range(13, 25)),
)


def main():
    parser = argparse.ArgumentParser(
        description="Solve only the 17 additional Atlas FEM cases."
    )
    parser.add_argument(
        "--points",
        type=int,
        default=100,
        help="Number of T/Tv points; default is 100.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recalculate an additional case if its NPZ result exists.",
    )
    arguments = parser.parse_args()

    if arguments.points < 2:
        raise ValueError("--points must be at least 2.")

    project_directory = Path(__file__).resolve().parent
    parametric_solver = project_directory / "FB2_parametric_solver.py"
    validation_solver = project_directory / "FB2_validation_solver.py"
    cache_root = project_directory / ".fenics_atlas_worker_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    worker_groups = [
        ADDITIONAL_CASES[worker_number::4]
        for worker_number in range(4)
    ]

    def run_worker(worker_number, case_names):
        worker_environment = os.environ.copy()
        worker_environment["XDG_CACHE_HOME"] = str(
            cache_root / f"worker_{worker_number + 1}"
        )

        for case_name in case_names:
            solver_path = (
                parametric_solver
                if case_name.startswith("TN")
                else validation_solver
            )
            command = [
                sys.executable,
                str(solver_path),
                "--case",
                case_name,
                "--points",
                str(arguments.points),
            ]
            if arguments.overwrite:
                command.append("--overwrite")

            print(
                f"Worker {worker_number + 1}: starting {case_name}",
                flush=True,
            )
            completed = subprocess.run(
                command,
                cwd=project_directory,
                env=worker_environment,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Worker {worker_number + 1}: {case_name} failed "
                    f"with return code {completed.returncode}."
                )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_worker, worker_number, case_names)
            for worker_number, case_names in enumerate(worker_groups)
        ]
        for future in futures:
            future.result()

    print("All 17 additional Atlas FEM cases are complete.")


if __name__ == "__main__":
    main()
