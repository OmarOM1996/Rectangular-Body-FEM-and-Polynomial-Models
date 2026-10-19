import argparse
import gc
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh


SURFACE = 9
BODY = 10
BOTTOM = 11
LEFT = 12
RIGHT = 13
LOWER = 14

GRAVITY = 9.81
WATER_DENSITY = 1000.0
WAVE_HEIGHT = 0.03
TAU_MIN = 0.25
TAU_MAX = 3.0
TAU_POINTS = 100


def wave_number(depth, period, gravity=9.81):
    """Calculate finite-depth wavenumber k using the existing approximation."""
    if depth <= 0.0:
        raise ValueError("Water depth must be positive.")

    if period <= 0.0:
        raise ValueError("Wave period must be positive.")

    omega = 2.0 * np.pi / period
    k0d = omega**2 * depth / gravity

    coefficients = np.array([
        0.00011,
        0.00039,
        0.00171,
        0.00654,
        0.02174,
        0.06320,
        0.16084,
        0.35550,
        0.66667,
        1.0,
    ])

    polynomial = np.polyval(coefficients, k0d)
    kd = np.sqrt(k0d**2 + k0d / polynomial)
    return kd / depth


def assemble_generalized_forces(
    domain,
    ds,
    pressure,
    surge_mode,
    heave_mode,
    pitch_mode,
):
    """Integrate pressure into surge force, heave force and pitch moment."""
    force_x_local = fem.assemble_scalar(
        fem.form(pressure * surge_mode * ds(BODY))
    )
    force_z_local = fem.assemble_scalar(
        fem.form(pressure * heave_mode * ds(BODY))
    )
    moment_y_local = fem.assemble_scalar(
        fem.form(pressure * pitch_mode * ds(BODY))
    )

    force_x = domain.comm.allreduce(force_x_local, op=MPI.SUM)
    force_z = domain.comm.allreduce(force_z_local, op=MPI.SUM)
    moment_y = domain.comm.allreduce(moment_y_local, op=MPI.SUM)

    return np.array(
        (force_x, force_z, moment_y),
        dtype=PETSc.ScalarType,
    )


def solve_diffraction(
    domain,
    dx,
    ds,
    n,
    x,
    V,
    depth,
    period,
    wave_height,
    gravity,
    rho,
    surge_mode,
    heave_mode,
    pitch_mode,
    petsc_prefix,
):
    """Solve the fixed-body diffraction problem and return normalized excitation."""
    wave_amplitude = wave_height / 2.0
    omega = 2.0 * np.pi / period
    k = wave_number(depth, period, gravity)
    omega2sug = fem.Constant(
        domain,
        PETSc.ScalarType(omega**2 / gravity),
    )

    phi_incident = (
        1j
        * gravity * wave_amplitude / omega
        * ufl.cosh(k * (x[1] + depth)) / np.cosh(k * depth)
        * ufl.exp(-1j * k * x[0])
    )

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    F = ufl.inner(ufl.nabla_grad(u), ufl.nabla_grad(v)) * dx
    BCsurf = omega2sug * ufl.inner(u, v) * ds(SURFACE)
    BCside = (
        -1j * k * ufl.inner(u, v) * ds(RIGHT)
        -1j * k * ufl.inner(u, v) * ds(LEFT)
    )
    BCbody = -ufl.inner(
        ufl.dot(ufl.grad(phi_incident), n),
        v,
    ) * ds(BODY)

    a_form = F - BCsurf - BCside
    L_form = BCbody

    problem = LinearProblem(
        a_form,
        L_form,
        bcs=[],
        petsc_options_prefix=petsc_prefix,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    phi_scattered = problem.solve()

    solver_reason = problem.solver.getConvergedReason()
    if solver_reason <= 0:
        raise RuntimeError(
            f"Diffraction solve failed at T={period} s; "
            f"PETSc reason={solver_reason}"
        )

    phi_total = phi_incident + phi_scattered
    pressure = -1j * rho * omega * phi_total
    excitation = assemble_generalized_forces(
        domain,
        ds,
        pressure,
        surge_mode,
        heave_mode,
        pitch_mode,
    )

    return excitation / wave_amplitude


def solve_radiation(
    domain,
    dx,
    ds,
    V,
    depth,
    period,
    gravity,
    rho,
    body_velocity_normal,
    surge_mode,
    heave_mode,
    pitch_mode,
    petsc_prefix,
):
    """Solve one unit-displacement radiation problem and return its force column."""
    omega = 2.0 * np.pi / period
    k = wave_number(depth, period, gravity)
    omega2sug = fem.Constant(
        domain,
        PETSc.ScalarType(omega**2 / gravity),
    )

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    F = ufl.inner(ufl.nabla_grad(u), ufl.nabla_grad(v)) * dx
    BCsurf = omega2sug * ufl.inner(u, v) * ds(SURFACE)
    BCside = (
        -1j * k * ufl.inner(u, v) * ds(RIGHT)
        -1j * k * ufl.inner(u, v) * ds(LEFT)
    )
    BCbody = ufl.inner(body_velocity_normal, v) * ds(BODY)

    a_form = F - BCsurf - BCside
    L_form = BCbody

    problem = LinearProblem(
        a_form,
        L_form,
        bcs=[],
        petsc_options_prefix=petsc_prefix,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    phi_radiation = problem.solve()

    solver_reason = problem.solver.getConvergedReason()
    if solver_reason <= 0:
        raise RuntimeError(
            f"Radiation solve failed at T={period} s; "
            f"PETSc reason={solver_reason}"
        )

    pressure = -1j * rho * omega * phi_radiation
    return assemble_generalized_forces(
        domain,
        ds,
        pressure,
        surge_mode,
        heave_mode,
        pitch_mode,
    )


def coefficient_scales(a, d, rho):
    """Return the Figure-3-compatible scales for modes 1, 3 and 5."""
    translation_mass_scale = 2.0 * rho * a * d
    pitch_inertia_scale = 8.0 * rho * a**3 * d
    mode_scales = np.array((
        translation_mass_scale,
        translation_mass_scale,
        pitch_inertia_scale,
    ))
    return np.sqrt(np.outer(mode_scales, mode_scales))


def save_curve(
    x,
    y,
    xlabel,
    ylabel,
    title,
    output_path,
    x_limits=None,
):
    """Save one curve in one figure."""
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(x, y, linewidth=2.0)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    if x_limits is not None:
        axis.set_xlim(x_limits)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_hydrodynamic_summary(
    x_values,
    added_mass_nondimensional,
    radiation_damping_nondimensional,
    xlabel,
    output_path,
    figure_title,
    x_limits=None,
    use_gesraha_labels=False,
):
    """Save the four principal and surge-pitch coefficient pairs."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    if use_gesraha_labels:
        panels = (
            (0, 0, "A11*", "B11*", "Surge: Gesraha A11, B11"),
            (1, 1, "A33*", "B33*", "Heave: Gesraha A22, B22"),
            (0, 2, "A15*", "B15*", "Surge-pitch: Gesraha A13, B13"),
            (2, 2, "A55*", "B55*", "Pitch: Gesraha A33, B33"),
        )
    else:
        panels = (
            (0, 0, "A11*", "B11*", "Surge: A11, B11"),
            (1, 1, "A33*", "B33*", "Heave: A33, B33"),
            (0, 2, "A15*", "B15*", "Surge-pitch: A15, B15"),
            (2, 2, "A55*", "B55*", "Pitch: A55, B55"),
        )

    for axis, panel in zip(axes.flat, panels):
        row, column, added_mass_label, damping_label, title = panel
        axis.plot(
            x_values,
            added_mass_nondimensional[:, row, column],
            linewidth=2.0,
            label=added_mass_label,
        )
        axis.plot(
            x_values,
            radiation_damping_nondimensional[:, row, column],
            linewidth=2.0,
            label=damping_label,
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Nondimensional coefficient [-]")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.legend()
        if x_limits is not None:
            axis.set_xlim(x_limits)

    figure.suptitle(figure_title, fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

#  #the organizer: Receives all the results for one case, Selects which columns 
# and matrix entries to plot, and Calculates excitation amplitude and phase. 
# Selects the correct labels, units and filenames.
def plot_case_results(        
    case_directory,
    fingerprint,
    a,
    d,
    rho,
    gravity,
    tau,
    ka,
    excitation,
    added_mass,
    radiation_damping,
    added_mass_nondimensional,
    radiation_damping_nondimensional,
):
    """Generate all case-specific excitation and coefficient figures."""
    figures_directory = case_directory / "figures"
    excitation_directory = figures_directory / "excitation"
    dimensional_directory = figures_directory / "dimensional_coefficients"
    nondimensional_directory = figures_directory / "nondimensional_coefficients"
    nondimensional_tau_directory = nondimensional_directory / "tau"
    nondimensional_ka_directory = nondimensional_directory / "ka"
    summary_directory = figures_directory / "hydrodynamic_summary"

    excitation_directory.mkdir(parents=True, exist_ok=True)
    dimensional_directory.mkdir(parents=True, exist_ok=True)
    nondimensional_tau_directory.mkdir(parents=True, exist_ok=True)
    nondimensional_ka_directory.mkdir(parents=True, exist_ok=True)
    summary_directory.mkdir(parents=True, exist_ok=True)

    xlabel = "Nondimensional period T/Tv [-]"
    mode_names = ("surge", "heave", "pitch")
    mode_titles = ("Surge", "Heave", "Pitch")
    amplitude_units = ("N/m²", "N/m²", "N/m")

    for mode_index, mode_name in enumerate(mode_names):
        save_curve(
            tau,
            np.abs(excitation[:, mode_index]),
            xlabel,
            f"Normalized excitation amplitude [{amplitude_units[mode_index]}]",
            f"{mode_titles[mode_index]} excitation amplitude",
            excitation_directory
            / f"{fingerprint}_excitation_{mode_name}_amplitude.png",
        )
        save_curve(
            tau,
            np.degrees(np.unwrap(np.angle(excitation[:, mode_index]))),
            xlabel,
            "Phase [deg]",
            f"{mode_titles[mode_index]} excitation phase",
            excitation_directory
            / f"{fingerprint}_excitation_{mode_name}_phase.png",
        )

    mode_numbers = (1, 3, 5)

    for row, force_mode in enumerate(mode_numbers):
        for column, motion_mode in enumerate(mode_numbers):
            if force_mode in (1, 3) and motion_mode in (1, 3):
                added_mass_unit = "kg/m"
                damping_unit = "kg/(m·s)"
            elif force_mode == 5 and motion_mode == 5:
                added_mass_unit = "kg·m"
                damping_unit = "kg·m/s"
            else:
                added_mass_unit = "kg"
                damping_unit = "kg/s"

            added_mass_name = f"A{force_mode}{motion_mode}"
            damping_name = f"B{force_mode}{motion_mode}"

            save_curve(
                tau,
                added_mass[:, row, column],
                xlabel,
                f"{added_mass_name} [{added_mass_unit}]",
                f"Added-mass coefficient {added_mass_name}",
                dimensional_directory
                / f"{fingerprint}_added_mass_{added_mass_name}.png",
            )
            save_curve(
                tau,
                radiation_damping[:, row, column],
                xlabel,
                f"{damping_name} [{damping_unit}]",
                f"Radiation-damping coefficient {damping_name}",
                dimensional_directory
                / f"{fingerprint}_radiation_damping_{damping_name}.png",
            )
            save_curve(
                tau,
                added_mass_nondimensional[:, row, column],
                xlabel,
                f"Nondimensional {added_mass_name} [-]",
                f"Nondimensional added-mass coefficient {added_mass_name}",
                nondimensional_tau_directory
                / (
                    f"{fingerprint}_nondimensional_added_mass_"
                    f"{added_mass_name}_vs_tau.png"
                ),
            )
            save_curve(
                tau,
                radiation_damping_nondimensional[:, row, column],
                xlabel,
                f"Nondimensional {damping_name} [-]",
                f"Nondimensional radiation-damping coefficient {damping_name}",
                nondimensional_tau_directory
                / (
                    f"{fingerprint}_nondimensional_"
                    f"radiation_damping_{damping_name}_vs_tau.png"
                ),
            )
            save_curve(
                ka,
                added_mass_nondimensional[:, row, column],
                "Nondimensional wavenumber ka [-]",
                f"Nondimensional {added_mass_name} [-]",
                f"Nondimensional added-mass coefficient {added_mass_name}",
                nondimensional_ka_directory
                / (
                    f"{fingerprint}_nondimensional_added_mass_"
                    f"{added_mass_name}_vs_ka.png"
                ),
                (0.0, 4.0),
            )
            save_curve(
                ka,
                radiation_damping_nondimensional[:, row, column],
                "Nondimensional wavenumber ka [-]",
                f"Nondimensional {damping_name} [-]",
                f"Nondimensional radiation-damping coefficient {damping_name}",
                nondimensional_ka_directory
                / (
                    f"{fingerprint}_nondimensional_"
                    f"radiation_damping_{damping_name}_vs_ka.png"
                ),
                (0.0, 4.0),
            )

    save_hydrodynamic_summary(
        tau,
        added_mass_nondimensional,
        radiation_damping_nondimensional,
        "Nondimensional period T/Tv [-]",
        summary_directory
        / f"{fingerprint}_hydrodynamic_summary_vs_tau.png",
        "Gesraha Figure 3 normalization versus T/Tv",
    )
    save_hydrodynamic_summary(
        ka,
        added_mass_nondimensional,
        radiation_damping_nondimensional,
        "Nondimensional wavenumber ka [-]",
        summary_directory
        / f"{fingerprint}_hydrodynamic_summary_vs_ka.png",
        "Gesraha Figure 3 normalization versus ka",
        (0.0, 4.0),
    )

    if fingerprint.startswith("GV"):
        comparison_directory = figures_directory / "gesraha_comparison"
        comparison_directory.mkdir(parents=True, exist_ok=True)

        gesraha_translation_scale = 2.0 * rho * a * d
        gesraha_pitch_scale = 8.0 * rho * a**3 * d
        gesraha_mode_scales = np.array((
            gesraha_translation_scale,
            gesraha_translation_scale,
            gesraha_pitch_scale,
        ))
        gesraha_scales = np.sqrt(
            np.outer(gesraha_mode_scales, gesraha_mode_scales)
        )
        gesraha_added_mass = added_mass / gesraha_scales[None, :, :]
        gesraha_damping = radiation_damping / (
            np.sqrt(gravity / a) * gesraha_scales[None, :, :]
        )
        gesraha_added_mass[:, 0, 2] *= -1.0
        gesraha_added_mass[:, 2, 0] *= -1.0
        gesraha_damping[:, 0, 2] *= -1.0
        gesraha_damping[:, 2, 0] *= -1.0

        save_hydrodynamic_summary(
            ka,
            gesraha_added_mass,
            gesraha_damping,
            "Nondimensional wavenumber ka [-]",
            comparison_directory
            / f"{fingerprint}_gesraha_figure3_vs_ka.png",
            "Gesraha Figure 3 normalization versus ka",
            (0.0, 4.0),
            True,
        )

##Its purpose is to save the results for one TN case in CSV table
def save_case_csv(  
    output_path,
    tau,
    periods,
    frequencies,
    omega,
    wavenumbers,
    ka,
    excitation,
    added_mass,
    radiation_damping,
    added_mass_nondimensional,
    radiation_damping_nondimensional,
):
    """Save a human-readable table in addition to the complete NPZ data."""
    columns = [
        tau,
        periods,
        frequencies,
        omega,
        wavenumbers,
        ka,
    ]
    headers = [
        "T_over_Tv",
        "period_s",
        "frequency_hz",
        "omega_rad_s",
        "wavenumber_1_m",
        "ka",
    ]

    mode_numbers = (1, 3, 5)
    for column, mode_number in enumerate(mode_numbers):
        values = excitation[:, column]
        columns.extend((
            values.real,
            values.imag,
            np.abs(values),
            np.degrees(np.unwrap(np.angle(values))),
        ))
        headers.extend((
            f"F{mode_number}_real",
            f"F{mode_number}_imag",
            f"F{mode_number}_amplitude",
            f"F{mode_number}_phase_deg",
        ))

    for name, values in (
        ("A", added_mass),
        ("B", radiation_damping),
        ("Astar", added_mass_nondimensional),
        ("Bstar", radiation_damping_nondimensional),
    ):
        for row, force_mode in enumerate(mode_numbers):
            for column, motion_mode in enumerate(mode_numbers):
                columns.append(values[:, row, column])
                headers.append(f"{name}{force_mode}{motion_mode}")

    np.savetxt(
        output_path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
    )


def solve_case(case_directory, metadata, tau_points):
    """Run one complete diffraction and radiation sweep."""
#Read the case parameters
    fingerprint = metadata["fingerprint"]
    a = float(metadata["a"])
    d = float(metadata["d"])
    depth = float(metadata["h"])
    mesh_path = case_directory / metadata["mesh_file"]
## import the Gmsh mesh into DOLFINx.
    mesh_data = gmsh.read_from_msh(
        str(mesh_path),
        MPI.COMM_WORLD,
        rank=0,
        gdim=2,
    )
    domain = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
# define integration measures
#dx is used for integration inside the fluid domain.
    dx = ufl.Measure(
        "dx",
        domain=domain,
        subdomain_data=cell_tags,
    )
    #ds is used for boundary integration, such as: ds(BODY), ds(SURFACE),ds(LEFT)..etc,
    ds = ufl.Measure(
        "ds",
        domain=domain,
        subdomain_data=facet_tags,
    )
## Geometry variables
    n = ufl.FacetNormal(domain)
    x = ufl.SpatialCoordinate(domain)
#The pitch reference is 0,0 so the pitch is defined about the top center.
    x_ref = 0.0
    z_ref = 0.0
## Define the body modes
    surge_mode = n[0]
    heave_mode = n[1]
    pitch_mode = (
        (x[1] - z_ref) * n[0]
        - (x[0] - x_ref) * n[1]
    )
# Create the FEM function space
    V = fem.functionspace(domain, ("Lagrange", 1))
#Create the period range
    Tv = 2.0 * np.pi * np.sqrt((d + 0.7 * a) / GRAVITY)
    ##
    # here we have tau= T/Tv for 100 values, it can change in the terminal
    tau = np.linspace(TAU_MIN, TAU_MAX, tau_points)
    periods = tau * Tv
    frequencies = 1.0 / periods
    omega = 2.0 * np.pi / periods
## The function calls wave_number() for every period:
    wavenumbers = np.array([
        wave_number(depth, period, GRAVITY)
        for period in periods
    ])
    ka = wavenumbers * a
# Prepare empty result arrays
    excitation = np.zeros(
        (tau_points, 3),
        dtype=PETSc.ScalarType,
    )
    #The radiation results are prepared as:
    added_mass = np.zeros((tau_points, 3, 3))
    radiation_damping = np.zeros((tau_points, 3, 3))

    start_time = MPI.Wtime()
## Loop over the periods
    for case_number, period in enumerate(periods):
        ## For every period:
        current_omega = omega[case_number]
        # Define the three body velocities
        surge_body_velocity = 1j * current_omega * surge_mode
        heave_body_velocity = 1j * current_omega * heave_mode
        pitch_body_velocity = 1j * current_omega * pitch_mode
        prefix_base = f"{fingerprint}_{case_number}_"


        # Solve diffraction
        excitation[case_number, :] = solve_diffraction(
            domain,
            dx,
            ds,
            n,
            x,
            V,
            depth,
            period,
            WAVE_HEIGHT,
            GRAVITY,
            WATER_DENSITY,
            surge_mode,
            heave_mode,
            pitch_mode,
            f"{prefix_base}diffraction_",
        )

        ## Solve the three radiation problems

        radiation_force_surge = solve_radiation(
            domain,
            dx,
            ds,
            V,
            depth,
            period,
            GRAVITY,
            WATER_DENSITY,
            surge_body_velocity,
            surge_mode,
            heave_mode,
            pitch_mode,
            f"{prefix_base}radiation_surge_",
        )
        radiation_force_heave = solve_radiation(
            domain,
            dx,
            ds,
            V,
            depth,
            period,
            GRAVITY,
            WATER_DENSITY,
            heave_body_velocity,
            surge_mode,
            heave_mode,
            pitch_mode,
            f"{prefix_base}radiation_heave_",
        )
        radiation_force_pitch = solve_radiation(
            domain,
            dx,
            ds,
            V,
            depth,
            period,
            GRAVITY,
            WATER_DENSITY,
            pitch_body_velocity,
            surge_mode,
            heave_mode,
            pitch_mode,
            f"{prefix_base}radiation_pitch_",
        )

        ## They are combned as columns (Q=[Q11 Q13 Q15
        #                                  Q31 Q33 Q35
        #                                  Q51 Q53 Q55])

        radiation_force_matrix = np.column_stack((
            radiation_force_surge,
            radiation_force_heave,
            radiation_force_pitch,
        ))
        # Calculate added mass and damping
        added_mass[case_number, :, :] = (
            radiation_force_matrix.real / current_omega**2
        )
        radiation_damping[case_number, :, :] = (
            -radiation_force_matrix.imag / current_omega
        )

        if domain.comm.rank == 0 and (
            case_number == 0
            or (case_number + 1) % 10 == 0
            or case_number + 1 == tau_points
        ):
            print(
                f"{fingerprint}: frequency point "
                f"{case_number + 1}/{tau_points}"
            )

    domain.comm.Barrier()
    elapsed_seconds = MPI.Wtime() - start_time
## Nondimensionalize the matrices (After all 100 periods are finished:)
    scales = coefficient_scales(a, d, WATER_DENSITY)
    added_mass_nondimensional = (
        added_mass / scales[None, :, :]
    )
    radiation_damping_nondimensional = (
        radiation_damping
        / (np.sqrt(GRAVITY / a) * scales[None, :, :])
    )

    if domain.comm.rank == 0:
        result_path = case_directory / f"{fingerprint}_results.npz"
        csv_path = case_directory / f"{fingerprint}_results.csv"

        #Save the NPZ archive
        np.savez_compressed(
            result_path,
            fingerprint=np.array(fingerprint),
            a=np.array(a),
            d=np.array(d),
            h=np.array(depth),
            gravity=np.array(GRAVITY),
            rho=np.array(WATER_DENSITY),
            wave_height=np.array(WAVE_HEIGHT),
            Tv=np.array(Tv),
            T_over_Tv=tau,
            periods=periods,
            frequencies=frequencies,
            omega=omega,
            wavenumbers=wavenumbers,
            ka=ka,
            excitation=excitation,
            added_mass=added_mass,
            radiation_damping=radiation_damping,
            added_mass_nondimensional=added_mass_nondimensional,
            radiation_damping_nondimensional=radiation_damping_nondimensional,
        )

        #Save the CSV file
        save_case_csv(
            csv_path,
            tau,
            periods,
            frequencies,
            omega,
            wavenumbers,
            ka,
            excitation,
            added_mass,
            radiation_damping,
            added_mass_nondimensional,
            radiation_damping_nondimensional,
        )

        #Create all figures
        plot_case_results(
            case_directory,
            fingerprint,
            a,
            d,
            WATER_DENSITY,
            GRAVITY,
            tau,
            ka,
            excitation,
            added_mass,
            radiation_damping,
            added_mass_nondimensional,
            radiation_damping_nondimensional,
        )

        #Update the metadata (add some information, Tv, frequency range,
        #  number of points,calculation time, completion date ...etc)
        #metadata means information describing one TN case. It is read from
        #  that case’s case_parameters.json file
        metadata.update({
            "gravity": GRAVITY,
            "rho": WATER_DENSITY,
            "wave_height": WAVE_HEIGHT,
            "Tv": Tv,
            "T_over_Tv_min": TAU_MIN,
            "T_over_Tv_max": TAU_MAX,
            "T_over_Tv_points": tau_points,
            "result_npz": result_path.name,
            "result_csv": csv_path.name,
            "elapsed_seconds": elapsed_seconds,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
        (case_directory / "case_parameters.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        print(
            f"{fingerprint}: completed in "
            f"{elapsed_seconds:.1f} seconds"
        )

        ## these two coming line removes the completed mesh from memory before the next TN case is loaded.
    del domain
    gc.collect()


def selected_case_directories(project_directory, requested_case):
    """Return TN cases and the separate Gesraha validation case."""
    cases_directory = project_directory / "FB2_cases"
    validation_directory = project_directory / "FB2_Gesraha_validation"

    case_directories = sorted(
        (
            path
            for path in cases_directory.glob("TN*_d*_da*")
            if path.is_dir()
        ),
    )
    validation_directories = sorted(
        (
            path
            for path in validation_directory.glob("GV*_d*_da*")
            if path.is_dir()
        ),
    )
    all_case_directories = case_directories + validation_directories

    if requested_case is None:
        return all_case_directories

    requested_case = requested_case.upper()
    if requested_case.isdigit():
        requested_case = f"TN{int(requested_case):02d}"
    elif requested_case.startswith("TN"):
        requested_case = f"TN{int(requested_case[2:]):02d}"
    elif requested_case.startswith("GV"):
        requested_case = f"GV{int(requested_case[2:]):02d}"

    # Find the matching folder    
    matches = [
        path
        for path in all_case_directories
        if path.name.startswith(f"{requested_case}_")
    ]

    # Check that it exists (just for safety, I will use it because i have many steps so there maybe an issue)
    if not matches:
        raise ValueError(f"No case folder found for {requested_case}.")

    ## Return the selected folder
    return matches


def run_parallel_cases(project_directory, points, overwrite):
    """Run four independent TN/GV workers with separate FEniCS caches."""
    case_names = [
        f"TN{case_number:02d}"
        for case_number in range(1, 33)
    ]
    case_names.append("GV01")
    worker_case_groups = [
        case_names[worker_number::4]
        for worker_number in range(4)
    ]
    solver_path = Path(__file__).resolve()
    cache_root = project_directory / ".fenics_worker_cache"
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
            executor.submit(
                run_worker,
                worker_number,
                worker_cases,
            )
            for worker_number, worker_cases
            in enumerate(worker_case_groups)
        ]
        for future in futures:
            future.result()


def main():
    parser = argparse.ArgumentParser(
        description="Solve all FB2 parametric hydrodynamic cases."
    )

    #Select one case (This allows: python FB2_parametric_solver.py --case TN10, without --case all cases are selected)
    parser.add_argument(
        "--case",
        help="Optional single case, for example TN10, 10, or GV01.",
    )
    #Force recalculation Without --overwrite, an existing result is skipped.
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recalculate cases whose NPZ result already exists.",
    )
    #Select the number of points, this allows: python FB2_parametric_solver.py --points 50
    parser.add_argument(
        "--points",
        type=int,
        default=TAU_POINTS,
        help="Number of T/Tv points; default is 100.",
    )
    #Read the supplied options
    arguments = parser.parse_args()

    #Check the number of points
    if arguments.points < 2:
        raise ValueError("--points must be at least 2.")

    #Find the project folder and select the requested case folders
    project_directory = Path(__file__).resolve().parent

    #Without --case, distribute all TN and GV cases over four CPU workers.
    if arguments.case is None:
        if MPI.COMM_WORLD.size != 1:
            raise RuntimeError(
                "Run the automatic four-case launcher with ordinary Python, "
                "not mpirun."
            )
        run_parallel_cases(
            project_directory,
            arguments.points,
            arguments.overwrite,
        )
        if MPI.COMM_WORLD.rank == 0:
            print("All parametric and validation cases are complete.")
        return

    case_directories = selected_case_directories(
        project_directory,
        arguments.case,
    )

    #Loop over the selected cases
    for case_directory in case_directories:
        #Read the metadata
        metadata_path = case_directory / "case_parameters.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        #Determine the expected result file
        fingerprint = metadata["fingerprint"]
        result_path = case_directory / f"{fingerprint}_results.npz"

        #Decide whether to skip the case
        skip_case = (
            result_path.exists()
            and not arguments.overwrite
        )
        skip_case = MPI.COMM_WORLD.bcast(
            skip_case if MPI.COMM_WORLD.rank == 0 else None,
            root=0,
        )
        #Skip an existing result
        if skip_case:
            if MPI.COMM_WORLD.rank == 0:
                print(f"{fingerprint}: existing result found; skipped")
            continue

        #Solve the case
        solve_case(
            case_directory,
            metadata,
            arguments.points,
        )

    if MPI.COMM_WORLD.rank == 0:
        print("Requested parametric and validation cases are complete.")


if __name__ == "__main__":
    main()
