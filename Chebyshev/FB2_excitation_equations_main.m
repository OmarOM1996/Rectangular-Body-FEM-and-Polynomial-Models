function [TNStatistics, VNStatistics] = FB2_excitation_equations_main()
%% Excitation analytical-model fitting and independent validation
% TN01--TN37 are used for fitting. VN01--VN24 are used only for validation.
% Existing FEM MAT files are read; the FEM calculation is not repeated.

tic;
close all;

%% USER CONTROLS

CREATE_TN_PER_CASE_COMPARISONS = true;
CREATE_TN_OVERVIEWS = true;
CREATE_VN_PER_CASE_COMPARISONS = true;
CREATE_VN_OVERVIEWS = true;

% Additional weight assigned to the excitation-force magnitude.
% Set to zero to recover fitting of the real and imaginary parts only.
MAGNITUDE_WEIGHT = 1.0;

% Rows: Re(F1*), Im(F1*), Re(F3*), Im(F3*), Re(F5*), Im(F5*).
% Leave empty to calculate one fixed scale for each component from all
% TN FEM and equation curves. The same scales are also used for VN plots.
EXCITATION_Y_LIMITS = [];

%% PROJECT FOLDERS

projectDirectory = fileparts(mfilename('fullpath'));
dataDirectory = fullfile(projectDirectory, 'data_mat');
validationDirectory = fullfile(projectDirectory, 'validation_cases');
equationDirectory = fullfile(projectDirectory, 'equations');
excitationDirectory = fullfile(projectDirectory, 'excitation_equations');
TNOutputDirectory = fullfile(excitationDirectory, 'TN_FEM_equation_comparisons');
VNOutputDirectory = fullfile(excitationDirectory, 'VN_independent_validation');

addpath(projectDirectory);
addpath(equationDirectory);
addpath(excitationDirectory);

if ~isfolder(TNOutputDirectory), mkdir(TNOutputDirectory); end
if ~isfolder(VNOutputDirectory), mkdir(VNOutputDirectory); end

componentNames = {'F1_real', 'F1_imag', 'F3_real', 'F3_imag', ...
    'F5_real', 'F5_imag'};
componentLabels = {'Re(F_1^*)', 'Im(F_1^*)', 'Re(F_3^*)', ...
    'Im(F_3^*)', 'Re(F_5^*)', 'Im(F_5^*)'};

%% READ AND NORMALIZE TN01--TN37

TNFiles = sortedResultFiles(dataDirectory, 'TN*_results.mat', 37, 'TN');
[tau, TNGeometries, TNFingerprints, TNFEM] = ...
    loadExcitationCases(TNFiles);

tauRange = [min(tau), max(tau)];
aOverHRange = [min(TNGeometries(:, 1)), max(TNGeometries(:, 1))];
dOverARange = [min(TNGeometries(:, 2)), max(TNGeometries(:, 2))];

%% FIT THE SIX EQUATIONS

termOrders = excitationTermOrders();
numberOfTNCases = size(TNFEM, 1);
numberOfTauPoints = numel(tau);

tauAll = repmat(tau, numberOfTNCases, 1);
aOverHAll = repelem(TNGeometries(:, 1), numberOfTauPoints);
dOverAAll = repelem(TNGeometries(:, 2), numberOfTauPoints);

x = scaleVariable(tauAll, tauRange);
y = scaleVariable(aOverHAll, aOverHRange);
z = scaleVariable(dOverAAll, dOverARange);
basis = chebyshevBasis(x, y, z, termOrders);
targets = reshape(permute(TNFEM, [2, 1, 3]), [], 6);

% Use the linear least-squares solution as the initial estimate.
initialCoefficientMatrix = basis\targets;

% Jointly fit the real part, imaginary part and resulting magnitude.
componentPairs = [1, 2; 3, 4; 5, 6];
coefficientMatrix = zeros(size(initialCoefficientMatrix));
optimizationOptions = optimoptions('lsqnonlin', ...
    'Display', 'iter', ...
    'MaxIterations', 500, ...
    'FunctionTolerance', 1.0e-10, ...
    'StepTolerance', 1.0e-10);

for modeNumber = 1:size(componentPairs, 1)
    realColumn = componentPairs(modeNumber, 1);
    imaginaryColumn = componentPairs(modeNumber, 2);
    initialPair = initialCoefficientMatrix(:, ...
        [realColumn, imaginaryColumn]);

    optimizedVector = lsqnonlin( ...
        @(coefficientVector) excitationJointResidual( ...
            coefficientVector, basis, targets(:, realColumn), ...
            targets(:, imaginaryColumn), MAGNITUDE_WEIGHT), ...
        initialPair(:), [], [], optimizationOptions);

    optimizedPair = reshape(optimizedVector, size(basis, 2), 2);
    coefficientMatrix(:, realColumn) = optimizedPair(:, 1);
    coefficientMatrix(:, imaginaryColumn) = optimizedPair(:, 2);
end

TNPredictedTargets = basis*coefficientMatrix;
TNEquation = permute(reshape(TNPredictedTargets, ...
    numberOfTauPoints, numberOfTNCases, 6), [2, 1, 3]);

fixedYLimits = calculateFixedLimits(targets, TNPredictedTargets);
if ~isempty(EXCITATION_Y_LIMITS)
    if ~isequal(size(EXCITATION_Y_LIMITS), [6, 2]) || ...
            any(EXCITATION_Y_LIMITS(:, 1) >= EXCITATION_Y_LIMITS(:, 2))
        error('EXCITATION_Y_LIMITS must be empty or a valid 6-by-2 matrix.');
    end
    fixedYLimits = EXCITATION_Y_LIMITS;
end

term_orders = termOrders;
coefficient_matrix = coefficientMatrix;
component_names = componentNames;
tau_range = tauRange;
a_over_h_range = aOverHRange;
d_over_a_range = dOverARange;
magnitude_weight = MAGNITUDE_WEIGHT;
normalization = ['F1*=F1/(rho*g*zeta_a*h), ', ...
    'F3*=F3/(rho*g*zeta_a*h), F5*=F5/(rho*g*zeta_a*h^2); ', ...
    'stored FEM excitation is already divided by zeta_a=H/2.'];
save(fullfile(excitationDirectory, ...
    'FB2_excitation_equation_coefficients.mat'), 'term_orders', ...
    'coefficient_matrix', 'component_names', 'tau_range', ...
    'a_over_h_range', 'd_over_a_range', 'magnitude_weight', ...
    'normalization');

writeCoefficientTable(excitationDirectory, termOrders, ...
    coefficientMatrix, componentNames);
writematrix(fixedYLimits, fullfile(TNOutputDirectory, ...
    'TN01_TN37_fixed_excitation_y_limits.csv'));

%% TN FITTING COMPARISONS AND STATISTICS

TNStatistics = calculateStatistics(TNFEM, TNEquation, ...
    TNFingerprints, componentNames);
writetable(TNStatistics.summary, fullfile(TNOutputDirectory, ...
    'TN01_TN37_excitation_fit_summary.csv'));
writetable(TNStatistics.byCase, fullfile(TNOutputDirectory, ...
    'TN01_TN37_excitation_fit_by_case.csv'));

if CREATE_TN_PER_CASE_COMPARISONS
    savePerCaseComparisons(tau, TNFEM, TNEquation, TNFingerprints, ...
        componentNames, componentLabels, fixedYLimits, TNOutputDirectory, ...
        'FEM', 'Analytical model');
end
if CREATE_TN_OVERVIEWS
    saveOverviewComparisons(tau, TNFEM, TNEquation, TNGeometries, ...
        componentNames, componentLabels, fixedYLimits, ...
        fullfile(TNOutputDirectory, 'comparison_overviews'), 'TN', ...
        'FEM', 'Analytical model');
end

%% INDEPENDENT VALIDATION WITH VN01--VN24

VNFiles = sortedValidationFiles(validationDirectory);
[VNTau, VNGeometries, VNFingerprints, VNFEM] = ...
    loadExcitationCases(VNFiles);
if max(abs(VNTau-tau)) > 1.0e-12
    error('TN and VN cases use different T/Tv points.');
end

VNEquation = zeros(size(VNFEM));
for caseNumber = 1:size(VNFEM, 1)
    Fstar = evaluate_FB2_excitation_equations(tau, ...
        VNGeometries(caseNumber, 1), VNGeometries(caseNumber, 2));
    VNEquation(caseNumber, :, :) = excitationComponents(Fstar);
end

VNStatistics = calculateStatistics(VNFEM, VNEquation, ...
    VNFingerprints, componentNames);
writetable(VNStatistics.summary, fullfile(VNOutputDirectory, ...
    'VN01_VN24_excitation_validation_summary.csv'));
writetable(VNStatistics.byCase, fullfile(VNOutputDirectory, ...
    'VN01_VN24_excitation_validation_by_case.csv'));

if CREATE_VN_PER_CASE_COMPARISONS
    savePerCaseComparisons(tau, VNFEM, VNEquation, VNFingerprints, ...
        componentNames, componentLabels, fixedYLimits, VNOutputDirectory, ...
        'FEM', 'Analytical model');
end
if CREATE_VN_OVERVIEWS
    saveOverviewComparisons(tau, VNFEM, VNEquation, VNGeometries, ...
        componentNames, componentLabels, fixedYLimits, ...
        fullfile(VNOutputDirectory, 'comparison_overviews'), 'VN', ...
        'FEM', 'Analytical model');
end

fprintf('\nTN01--TN37 excitation fit:\n');
disp(TNStatistics.summary);
fprintf('\nVN01--VN24 independent excitation validation:\n');
disp(VNStatistics.summary);
fprintf('Excitation workflow complete.\n');
toc;

end

%% LOCAL FUNCTIONS

function files = sortedResultFiles(directory, pattern, expected, prefix)
files = dir(fullfile(directory, pattern));
if numel(files) ~= expected
    error('Expected %d %s files, found %d.', expected, prefix, numel(files));
end
numbers = zeros(numel(files), 1);
for index = 1:numel(files)
    token = regexp(files(index).name, ['^', prefix, '(\d+)'], ...
        'tokens', 'once');
    numbers(index) = str2double(token{1});
end
[~, order] = sort(numbers);
files = files(order);
end

function files = sortedValidationFiles(validationDirectory)
directories = dir(fullfile(validationDirectory, 'VN*'));
directories = directories([directories.isdir]);
[~, order] = sort({directories.name});
directories = directories(order);
files = repmat(dir(fullfile(validationDirectory, directories(1).name, ...
    'FEM_results', '*_results.mat')), numel(directories), 1);
for index = 1:numel(directories)
    oneFile = dir(fullfile(validationDirectory, directories(index).name, ...
        'FEM_results', '*_results.mat'));
    if numel(oneFile) ~= 1
        error('%s must contain exactly one FEM result MAT file.', ...
            directories(index).name);
    end
    files(index) = oneFile;
end
end

function [tau, geometries, fingerprints, curves] = loadExcitationCases(files)
numberOfCases = numel(files);
first = load(fullfile(files(1).folder, files(1).name));
tau = first.T_over_Tv(:);
curves = zeros(numberOfCases, numel(tau), 6);
geometries = zeros(numberOfCases, 2);
fingerprints = strings(numberOfCases, 1);

for index = 1:numberOfCases
    result = load(fullfile(files(index).folder, files(index).name));
    if max(abs(result.T_over_Tv(:)-tau)) > 1.0e-12
        error('%s uses different T/Tv points.', files(index).name);
    end
    fingerprint = strtrim(string(result.fingerprint));
    fingerprints(index) = fingerprint(1);
    a = double(result.a); d = double(result.d); h = double(result.h);
    rho = double(result.rho); gravity = double(result.gravity);
    geometries(index, :) = [a/h, d/a];

    % Python stores excitation after division by zeta_a=H/2.
    storedExcitation = result.excitation;
    Fstar = storedExcitation ./ ...
        [rho*gravity*h, rho*gravity*h, rho*gravity*h^2];
    curves(index, :, :) = excitationComponents(Fstar);
end
end

function components = excitationComponents(Fstar)
components = [real(Fstar(:, 1)), imag(Fstar(:, 1)), ...
    real(Fstar(:, 2)), imag(Fstar(:, 2)), ...
    real(Fstar(:, 3)), imag(Fstar(:, 3))];
end

function scaled = scaleVariable(value, range)
scaled = 2*(value-range(1))/(range(2)-range(1))-1;
end

function limits = calculateFixedLimits(FEM, equation)
limits = zeros(size(FEM, 2), 2);
for component = 1:size(FEM, 2)
    minimum = min([FEM(:, component); equation(:, component)]);
    maximum = max([FEM(:, component); equation(:, component)]);
    padding = 0.05*max(maximum-minimum, eps);
    limits(component, :) = [minimum-padding, maximum+padding];
end
end

function statistics = calculateStatistics(FEM, equation, fingerprints, names)
numberOfCases = size(FEM, 1);
numberOfComponents = size(FEM, 3);
globalNRMSE = zeros(numberOfComponents, 1);
caseNRMSE = zeros(numberOfCases, numberOfComponents);
caseRelativeL2 = zeros(numberOfCases, numberOfComponents);

for component = 1:numberOfComponents
    allFEM = reshape(FEM(:, :, component), [], 1);
    allEquation = reshape(equation(:, :, component), [], 1);
    globalNRMSE(component) = sqrt(mean((allFEM-allEquation).^2))/ ...
        max(max(allFEM)-min(allFEM), eps);
    for caseNumber = 1:numberOfCases
        oneFEM = squeeze(FEM(caseNumber, :, component)).';
        oneEquation = squeeze(equation(caseNumber, :, component)).';
        error = oneFEM-oneEquation;
        caseNRMSE(caseNumber, component) = sqrt(mean(error.^2))/ ...
            max(max(oneFEM)-min(oneFEM), eps);
        caseRelativeL2(caseNumber, component) = ...
            norm(error)/max(norm(oneFEM), eps);
    end
end

statistics.summary = table(string(names(:)), 100*globalNRMSE, ...
    100*max(caseNRMSE, [], 1).', 100*mean(caseRelativeL2, 1).', ...
    'VariableNames', {'Component', 'Global_NRMSE_percent', ...
    'Worst_case_NRMSE_percent', 'Mean_relative_L2_percent'});

caseName = strings(numberOfCases*numberOfComponents, 1);
componentName = strings(size(caseName));
NRMSEPercent = zeros(size(caseName));
relativeL2Percent = zeros(size(caseName));
row = 0;
for caseNumber = 1:numberOfCases
    for component = 1:numberOfComponents
        row = row+1;
        caseName(row) = fingerprints(caseNumber);
        componentName(row) = names{component};
        NRMSEPercent(row) = 100*caseNRMSE(caseNumber, component);
        relativeL2Percent(row) = 100*caseRelativeL2(caseNumber, component);
    end
end
statistics.byCase = table(caseName, componentName, NRMSEPercent, ...
    relativeL2Percent, 'VariableNames', {'Case', 'Component', ...
    'NRMSE_percent', 'Relative_L2_percent'});
end

function savePerCaseComparisons(tau, FEM, equation, fingerprints, ...
        names, labels, limits, root, FEMLegend, equationLegend)
for caseNumber = 1:size(FEM, 1)
    caseDirectory = fullfile(root, char(fingerprints(caseNumber)));
    if ~isfolder(caseDirectory), mkdir(caseDirectory); end
    for component = 1:numel(names)
        figureHandle = figure('Visible', 'off', 'Color', 'white');
        plot(tau, squeeze(FEM(caseNumber, :, component)), ...
            'LineWidth', 2, 'DisplayName', FEMLegend);
        hold on;
        plot(tau, squeeze(equation(caseNumber, :, component)), '--', ...
            'LineWidth', 2, 'DisplayName', equationLegend);
        hold off; grid on; box on;
        xlim([min(tau), max(tau)]); ylim(limits(component, :));
        xlabel('T/T_v [-]'); ylabel([labels{component}, ' [-]']);
        legend('Location', 'best');
        exportgraphics(figureHandle, fullfile(caseDirectory, sprintf( ...
            '%s_FEM_vs_equation_%s.png', fingerprints(caseNumber), ...
            names{component})), 'Resolution', 180);
        close(figureHandle);
    end
end
end

function saveOverviewComparisons(tau, FEM, equation, geometries, ...
        names, labels, limits, outputDirectory, prefix, FEMLegend, equationLegend)
if ~isfolder(outputDirectory), mkdir(outputDirectory); end
numberOfCases = size(FEM, 1);
columns = 4;
rows = ceil(numberOfCases/columns);
for component = 1:numel(names)
    figureHandle = figure('Visible', 'off', 'Color', 'white', ...
        'Position', [50, 50, 1500, max(950, 235*rows)]);
    tiledlayout(rows, columns, 'TileSpacing', 'compact', 'Padding', 'compact');
    for caseNumber = 1:numberOfCases
        nexttile;
        plot(tau, squeeze(FEM(caseNumber, :, component)), ...
            'LineWidth', 1.5, 'DisplayName', FEMLegend);
        hold on;
        plot(tau, squeeze(equation(caseNumber, :, component)), '--', ...
            'LineWidth', 1.4, 'DisplayName', equationLegend);
        hold off; grid on; box on;
        xlim([min(tau), max(tau)]); ylim(limits(component, :));
        title(sprintf('a/h=%.3f, d/a=%.3f', ...
            geometries(caseNumber, 1), geometries(caseNumber, 2)), ...
            'FontSize', 7);
        if caseNumber > numberOfCases-columns
            xlabel('T/T_v [-]');
        end
        if mod(caseNumber-1, columns) == 0
            ylabel([labels{component}, ' [-]']);
        end
        if caseNumber == 1, legend('Location', 'best', 'FontSize', 7); end
    end
    exportgraphics(figureHandle, fullfile(outputDirectory, sprintf( ...
        '%s_FEM_vs_equation_%s01_to_%s%02d.png', names{component}, ...
        prefix, prefix, numberOfCases)), 'Resolution', 180);
    close(figureHandle);
end
end

function writeCoefficientTable(directory, orders, coefficients, names)
output = array2table([orders, coefficients], 'VariableNames', ...
    [{'tau_order', 'a_over_h_order', 'd_over_a_order'}, names]);
writetable(output, fullfile(directory, ...
    'FB2_excitation_equation_coefficients.csv'));
end

function residual = excitationJointResidual(coefficientVector, basis, ...
        targetReal, targetImaginary, magnitudeWeight)
numberOfTerms = size(basis, 2);
realCoefficients = coefficientVector(1:numberOfTerms);
imaginaryCoefficients = coefficientVector(numberOfTerms+1:end);

predictedReal = basis*realCoefficients;
predictedImaginary = basis*imaginaryCoefficients;
targetMagnitude = hypot(targetReal, targetImaginary);
predictedMagnitude = hypot(predictedReal, predictedImaginary);

residualScale = max(targetMagnitude)-min(targetMagnitude);
if residualScale < 1.0e-12
    residualScale = max(max(targetMagnitude), 1.0);
end

residual = [(predictedReal-targetReal)/residualScale; ...
    (predictedImaginary-targetImaginary)/residualScale; ...
    sqrt(magnitudeWeight)* ...
    (predictedMagnitude-targetMagnitude)/residualScale];
end
