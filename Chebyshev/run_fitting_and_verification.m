%% Fit and verify the paper's 109-term Chebyshev expressions.
% Uses the stored FEM data; does not rerun the FEM solver.
clear;
projectDirectory = fileparts(mfilename('fullpath'));
addpath(projectDirectory);
addpath(fullfile(projectDirectory, 'equations'));
addpath(fullfile(projectDirectory, 'excitation_equations'));

assert(size(hydrodynamicTermOrders(), 1) == 109);
assert(size(excitationTermOrders(), 1) == 109);
assert(numel(dir(fullfile(projectDirectory, 'data_mat', ...
    'TN*_results.mat'))) == 37);
assert(numel(dir(fullfile(projectDirectory, 'validation_cases', ...
    'VN*', 'FEM_results', 'VN*_results.mat'))) == 24);

% Fit radiation coefficients using TN01--TN37 only.
fitSummary = fit_and_compare_TN01_TN37( ...
    projectDirectory, [], false, false, "ALL");
clear evaluate_FB2_hydrodynamic_equations

% Evaluate the fitted model for the independent verification geometries.
generate_VN_equation_predictions();
validationSummary = import_and_compare_VN_FEM();

% Fit and independently verify the real and imaginary excitation components.
[excitationFit, excitationValidation] = FB2_excitation_equations_main();
clear evaluate_FB2_excitation_equations

disp(fitSummary);
disp(validationSummary);
disp(excitationFit.summary);
disp(excitationValidation.summary);
