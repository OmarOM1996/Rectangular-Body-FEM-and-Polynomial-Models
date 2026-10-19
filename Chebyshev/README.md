# Chebyshev scripts

1. Run the FEM scripts in the parent folder for the 37 TN and 24 VN cases.
2. Run `python export_FEM_inputs.py` (requires NumPy and SciPy).
3. In MATLAB, run `run_fitting_and_verification` from this folder.

MATLAB requires Optimization Toolbox for excitation fitting (`lsqnonlin`).
Keep the helper functions and subfolder structure unchanged. The existing
verification functions also generate diagnostic figures. Coefficient files
are produced during fitting; fit first before using the evaluation functions.