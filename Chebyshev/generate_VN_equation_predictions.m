function generate_VN_equation_predictions()
% Generate equation predictions for VN01--VN24.
% The frozen TN01--TN37 Chebyshev equations are evaluated without using
% any VN result in the fitting process.

tic;
close all;

projectDirectory = fileparts(mfilename('fullpath'));
equationDirectory = fullfile(projectDirectory, 'equations');
validationDirectory = fullfile(projectDirectory, 'validation_cases');
overviewDirectory = fullfile(validationDirectory, ...
    'equation_prediction_overviews');

addpath(projectDirectory);
addpath(equationDirectory);

coefficientDefinitions = {
    'A11', 'added mass', 1, 1;
    'A33', 'added mass', 2, 2;
    'A55', 'added mass', 3, 3;
    'A15', 'added mass', 1, 3;
    'B11', 'damping',    1, 1;
    'B33', 'damping',    2, 2;
    'B55', 'damping',    3, 3;
    'B15', 'damping',    1, 3
};

fixedLimitPath = fullfile(equationDirectory, ...
    'TN_FEM_equation_comparisons', ...
    'TN01_TN37_fixed_comparison_y_limits.csv');
fixedLimitTable = readtable(fixedLimitPath, 'TextType', 'string');
fixedYLimits = zeros(size(coefficientDefinitions, 1), 2);
for coefficient = 1:size(coefficientDefinitions, 1)
    name = string(coefficientDefinitions{coefficient, 1});
    row = fixedLimitTable.Coefficient == name;
    fixedYLimits(coefficient, :) = [ ...
        fixedLimitTable.Fixed_y_min(row), ...
        fixedLimitTable.Fixed_y_max(row)];
end

caseDirectories = dir(fullfile(validationDirectory, 'VN*'));
caseDirectories = caseDirectories([caseDirectories.isdir]);
caseNumbers = zeros(numel(caseDirectories), 1);
for index = 1:numel(caseDirectories)
    token = regexp(caseDirectories(index).name, '^VN(\d+)', ...
        'tokens', 'once');
    caseNumbers(index) = str2double(token{1});
end
[caseNumbers, order] = sort(caseNumbers);
caseDirectories = caseDirectories(order);
numberOfCases = numel(caseDirectories);
if numberOfCases ~= 24
    error('Expected 24 VN folders, found %d.', numberOfCases);
end

caseNames = strings(numberOfCases, 1);
aOverHColumn = zeros(numberOfCases, 1);
dOverAColumn = zeros(numberOfCases, 1);
hColumn = zeros(numberOfCases, 1);
aColumn = zeros(numberOfCases, 1);
dColumn = zeros(numberOfCases, 1);
predictionCurves = [];
tauReference = [];

for caseIndex = 1:numberOfCases
    fingerprint = caseDirectories(caseIndex).name;
    caseDirectory = fullfile(validationDirectory, fingerprint);
    FEMPath = fullfile(caseDirectory, 'FEM_results', ...
        sprintf('%s_results.mat', fingerprint));
    FEM = load(FEMPath);

    tau = FEM.T_over_Tv(:);
    a = double(FEM.a);
    d = double(FEM.d);
    h = double(FEM.h);
    aOverH = a/h;
    dOverA = d/a;
    caseID = sprintf('VN%02d', caseNumbers(caseIndex));

    if isempty(tauReference)
        tauReference = tau;
        predictionCurves = zeros(numberOfCases, numel(tau), ...
            size(coefficientDefinitions, 1));
    elseif max(abs(tau-tauReference)) > 1.0e-12
        error('%s uses different T/Tv points.', fingerprint);
    end

    caseNames(caseIndex) = string(fingerprint);
    aOverHColumn(caseIndex) = aOverH;
    dOverAColumn(caseIndex) = dOverA;
    hColumn(caseIndex) = h;
    aColumn(caseIndex) = a;
    dColumn(caseIndex) = d;

    [Astar, Bstar] = evaluate_FB2_hydrodynamic_equations( ...
        tau, aOverH, dOverA);

    predictionDirectory = fullfile(caseDirectory, 'equation_predictions');
    curveDirectory = fullfile(predictionDirectory, 'figures', ...
        'nondimensional_coefficients', 'tau');
    summaryDirectory = fullfile(predictionDirectory, 'figures', ...
        'hydrodynamic_summary');
    if ~isfolder(predictionDirectory)
        mkdir(predictionDirectory);
    end

    save(fullfile(predictionDirectory, ...
        sprintf('%s_equation_prediction.mat', fingerprint)), ...
        'fingerprint', 'caseID', 'a', 'd', 'h', 'aOverH', ...
        'dOverA', 'tau', 'Astar', 'Bstar');

    for coefficient = 1:size(coefficientDefinitions, 1)
        coefficientName = coefficientDefinitions{coefficient, 1};
        coefficientType = coefficientDefinitions{coefficient, 2};
        matrixRow = coefficientDefinitions{coefficient, 3};
        matrixColumn = coefficientDefinitions{coefficient, 4};
        if strcmp(coefficientType, 'added mass')
            values = squeeze(Astar(:, matrixRow, matrixColumn));
        else
            values = squeeze(Bstar(:, matrixRow, matrixColumn));
        end
        predictionCurves(caseIndex, :, coefficient) = values;
        saveSingleCurve(tau, values, 'T/T_v [-]', ...
            sprintf('%s^* [-]', coefficientName), '', ...
            fullfile(curveDirectory, sprintf( ...
            '%s_equation_%s_vs_tau.png', fingerprint, coefficientName)), ...
            [min(tau), max(tau)]);
    end

    saveSummary(tau, Astar, Bstar, 'T/T_v [-]', '', ...
        fullfile(summaryDirectory, sprintf( ...
        '%s_equation_hydrodynamic_summary.png', fingerprint)), ...
        [min(tau), max(tau)]);
    fprintf('%s: equation predictions complete\n', fingerprint);
end

validationCaseTable = table(caseNumbers, caseNames, aOverHColumn, ...
    dOverAColumn, hColumn, aColumn, dColumn, ...
    'VariableNames', {'CaseNumber', 'Fingerprint', 'a_over_h', ...
    'd_over_a', 'h_m', 'a_m', 'd_m'});
writetable(validationCaseTable, fullfile(validationDirectory, ...
    'VN_validation_case_table.csv'));

if ~isfolder(overviewDirectory)
    mkdir(overviewDirectory);
end
for coefficient = 1:size(coefficientDefinitions, 1)
    coefficientName = coefficientDefinitions{coefficient, 1};
    figureHandle = figure('Visible', 'off', 'Color', 'white', ...
        'Units', 'inches', 'Position', [0.2, 0.2, 24, 12]);
    tiledlayout(4, 6, 'TileSpacing', 'compact', 'Padding', 'compact');
    for caseIndex = 1:numberOfCases
        nexttile;
        plot(tauReference, ...
            squeeze(predictionCurves(caseIndex, :, coefficient)), ...
            'LineWidth', 1.8);
        xlim([min(tauReference), max(tauReference)]);
        ylim(fixedYLimits(coefficient, :));
        title(sprintf('a/h=%.3f, d/a=%.3f', ...
            aOverHColumn(caseIndex), ...
            dOverAColumn(caseIndex)), 'FontSize', 8);
        if caseIndex > 18
            xlabel('T/T_v [-]');
        end
        if mod(caseIndex-1, 6) == 0
            ylabel(sprintf('%s^* [-]', coefficientName));
        end
        grid on;
        box on;
    end
    exportgraphics(figureHandle, fullfile(overviewDirectory, sprintf( ...
        '%s_equation_predictions_VN01_to_VN24.png', coefficientName)), ...
        'Resolution', 220);
    close(figureHandle);
end

fprintf('\nAll VN equation predictions were saved in:\n%s\n', ...
    validationDirectory);
toc;
end
