function summaryTable = import_and_compare_VN_FEM()
% Import VN01--VN24 FEM data and compare it with the frozen equations.
% The MAT files are prepared from the original NPZ archives by Python.

tic;
close all;

projectDirectory = fileparts(mfilename('fullpath'));
validationDirectory = fullfile(projectDirectory, 'validation_cases');
overviewDirectory = fullfile(validationDirectory, 'comparison_overviews');
if ~isfolder(overviewDirectory)
    mkdir(overviewDirectory);
end

addpath(projectDirectory);
addpath(fullfile(projectDirectory, 'equations'));

coefficientDefinitions = {
    'A11', 'added_mass_nondimensional',             'Astar', 1, 1;
    'A33', 'added_mass_nondimensional',             'Astar', 2, 2;
    'A55', 'added_mass_nondimensional',             'Astar', 3, 3;
    'A15', 'added_mass_nondimensional',             'Astar', 1, 3;
    'B11', 'radiation_damping_nondimensional',      'Bstar', 1, 1;
    'B33', 'radiation_damping_nondimensional',      'Bstar', 2, 2;
    'B55', 'radiation_damping_nondimensional',      'Bstar', 3, 3;
    'B15', 'radiation_damping_nondimensional',      'Bstar', 1, 3
};

% Use the same fixed coefficient scales as the TN01--TN37 comparisons.
fixedLimitPath = fullfile(projectDirectory, 'equations', ...
    'TN_FEM_equation_comparisons', ...
    'TN01_TN37_fixed_comparison_y_limits.csv');
if ~isfile(fixedLimitPath)
    error(['The TN fixed-scale file was not found: %s\n' ...
        'Run FB2_Atlas_main.m before this validation script.'], ...
        fixedLimitPath);
end
fixedLimitTable = readtable(fixedLimitPath, 'TextType', 'string');
fixedYLimits = zeros(size(coefficientDefinitions, 1), 2);
for coefficient = 1:size(coefficientDefinitions, 1)
    coefficientName = string(coefficientDefinitions{coefficient, 1});
    limitRow = fixedLimitTable.Coefficient == coefficientName;
    if nnz(limitRow) ~= 1
        error('No unique fixed y-axis scale was found for %s.', ...
            coefficientName);
    end
    fixedYLimits(coefficient, :) = [ ...
        fixedLimitTable.Fixed_y_min(limitRow), ...
        fixedLimitTable.Fixed_y_max(limitRow)];
end

caseDirectories = dir(fullfile(validationDirectory, 'VN*'));
caseDirectories = caseDirectories([caseDirectories.isdir]);
[~, order] = sort({caseDirectories.name});
caseDirectories = caseDirectories(order);
numberOfCases = numel(caseDirectories);
if numberOfCases ~= 24
    error('Expected 24 VN folders, found %d.', numberOfCases);
end
numberOfCoefficients = size(coefficientDefinitions, 1);

FEMCurves = [];
equationCurves = [];
tauValues = [];
aOverHValues = zeros(numberOfCases, 1);
dOverAValues = zeros(numberOfCases, 1);
caseNames = strings(numberOfCases, 1);
caseNRMSE = zeros(numberOfCases, numberOfCoefficients);
caseRelativeL2 = zeros(numberOfCases, numberOfCoefficients);

for caseNumber = 1:numberOfCases
    fingerprint = caseDirectories(caseNumber).name;
    caseNames(caseNumber) = string(fingerprint);
    caseDirectory = fullfile(validationDirectory, fingerprint);
    predictionDirectory = fullfile(caseDirectory, 'equation_predictions');
    FEMDirectory = fullfile(caseDirectory, 'FEM_results');
    comparisonDirectory = fullfile(caseDirectory, 'comparison');
    FEMFigureDirectory = fullfile(FEMDirectory, 'figures');
    FEMCurveDirectory = fullfile(FEMFigureDirectory, ...
        'nondimensional_coefficients', 'tau');
    FEMSummaryDirectory = fullfile(FEMFigureDirectory, ...
        'hydrodynamic_summary');

    prediction = load(fullfile(predictionDirectory, ...
        sprintf('%s_equation_prediction.mat', fingerprint)));
    FEM = load(fullfile(FEMDirectory, sprintf('%s_results.mat', fingerprint)));

    tau = FEM.T_over_Tv(:);
    if max(abs(tau-prediction.tau(:))) > 1.0e-12
        error('%s: FEM and equation T/Tv points are different.', fingerprint);
    end

    if isempty(tauValues)
        tauValues = tau;
        FEMCurves = zeros(numberOfCases, numel(tau), numberOfCoefficients);
        equationCurves = zeros(numberOfCases, numel(tau), numberOfCoefficients);
    end

    aOverHValues(caseNumber) = prediction.aOverH;
    dOverAValues(caseNumber) = prediction.dOverA;

    % Recreate the FEM plots in MATLAB using the same eight coefficients.
    for coefficient = 1:numberOfCoefficients
        coefficientName = coefficientDefinitions{coefficient, 1};
        FEMField = coefficientDefinitions{coefficient, 2};
        equationField = coefficientDefinitions{coefficient, 3};
        row = coefficientDefinitions{coefficient, 4};
        column = coefficientDefinitions{coefficient, 5};

        FEMMatrix = FEM.(FEMField);
        equationMatrix = prediction.(equationField);
        FEMValues = squeeze(FEMMatrix(:, row, column));
        equationValues = squeeze(equationMatrix(:, row, column));
        FEMCurves(caseNumber, :, coefficient) = FEMValues;
        equationCurves(caseNumber, :, coefficient) = equationValues;

        saveSingleCurve(tau, FEMValues, 'T/T_v [-]', ...
            sprintf('%s^* [-]', coefficientName), ...
            sprintf('FEM %s^*', coefficientName), ...
            fullfile(FEMCurveDirectory, sprintf( ...
            '%s_FEM_%s_vs_tau.png', fingerprint, coefficientName)), []);

        comparisonPath = fullfile(comparisonDirectory, sprintf( ...
            '%s_FEM_vs_equation_%s.png', fingerprint, coefficientName));
        saveComparisonCurve(tau, FEMValues, equationValues, ...
            coefficientName, fixedYLimits(coefficient, :), comparisonPath);

        errorValues = FEMValues-equationValues;
        FEMRange = max(FEMValues)-min(FEMValues);
        if FEMRange > eps
            caseNRMSE(caseNumber, coefficient) = ...
                sqrt(mean(errorValues.^2))/FEMRange;
        else
            caseNRMSE(caseNumber, coefficient) = NaN;
        end
        caseRelativeL2(caseNumber, coefficient) = ...
            norm(errorValues)/max(norm(FEMValues), eps);
    end

    saveSummary(tau, FEM.added_mass_nondimensional, ...
        FEM.radiation_damping_nondimensional, 'T/T_v [-]', '', ...
        fullfile(FEMSummaryDirectory, sprintf( ...
        '%s_FEM_hydrodynamic_summary.png', fingerprint)), []);

    saveComparisonSummary(tau, FEM, prediction, coefficientDefinitions, ...
        fixedYLimits, fullfile( ...
        comparisonDirectory, sprintf( ...
        '%s_FEM_vs_equation_summary.png', fingerprint)));

    fprintf('%s: MATLAB FEM plots and comparisons complete\n', fingerprint);
end

% Save 4x6 comparison overviews and coefficient-level error statistics.
globalNRMSE = zeros(numberOfCoefficients, 1);
worstCaseNRMSE = zeros(numberOfCoefficients, 1);
meanRelativeL2 = zeros(numberOfCoefficients, 1);

for coefficient = 1:numberOfCoefficients
    coefficientName = coefficientDefinitions{coefficient, 1};
    allFEM = reshape(FEMCurves(:, :, coefficient), [], 1);
    allEquation = reshape(equationCurves(:, :, coefficient), [], 1);
    globalRange = max(allFEM)-min(allFEM);
    globalNRMSE(coefficient) = ...
        sqrt(mean((allFEM-allEquation).^2))/globalRange;
    worstCaseNRMSE(coefficient) = max(caseNRMSE(:, coefficient));
    meanRelativeL2(coefficient) = mean(caseRelativeL2(:, coefficient));

    figureHandle = figure('Visible', 'off', 'Color', 'white', ...
        'Units', 'inches', 'Position', [0.2, 0.2, 24, 12]);
    tiledlayout(4, 6, 'TileSpacing', 'compact', 'Padding', 'compact');
    for caseNumber = 1:numberOfCases
        nexttile;
        plot(tauValues, squeeze(FEMCurves(caseNumber, :, coefficient)), ...
            'LineWidth', 1.8, 'DisplayName', 'FEM');
        hold on;
        plot(tauValues, squeeze(equationCurves(caseNumber, :, coefficient)), ...
            '--', 'LineWidth', 1.6, 'DisplayName', 'Analytical model');
        hold off;
        title(sprintf('a/h=%.3f, d/a=%.3f', ...
            aOverHValues(caseNumber), dOverAValues(caseNumber)), ...
            'FontSize', 8);
        xlim([min(tauValues), max(tauValues)]);
        ylim(fixedYLimits(coefficient, :));
        if caseNumber > 18
            xlabel('T/T_v [-]');
        end
        if mod(caseNumber-1, 6) == 0
            ylabel(sprintf('%s^* [-]', coefficientName));
        end
        grid on;
        box on;
        if caseNumber == 1
            legend('Location', 'best');
        end
    end
    exportgraphics(figureHandle, fullfile(overviewDirectory, sprintf( ...
        '%s_FEM_vs_equation_VN01_to_VN24.png', coefficientName)), ...
        'Resolution', 220);
    close(figureHandle);
end

coefficientNames = string(coefficientDefinitions(:, 1));
summaryTable = table(coefficientNames, 100*globalNRMSE, ...
    100*worstCaseNRMSE, 100*meanRelativeL2, ...
    'VariableNames', {'Coefficient', 'Global_NRMSE_percent', ...
    'Worst_case_NRMSE_percent', 'Mean_relative_L2_percent'});
writetable(summaryTable, fullfile(validationDirectory, ...
    'VN_equation_validation_summary.csv'));

caseErrorTable = table;
for caseNumber = 1:numberOfCases
    for coefficient = 1:numberOfCoefficients
        newRow = table(caseNames(caseNumber), ...
            string(coefficientDefinitions{coefficient, 1}), ...
            100*caseNRMSE(caseNumber, coefficient), ...
            100*caseRelativeL2(caseNumber, coefficient), ...
            'VariableNames', {'Fingerprint', 'Coefficient', ...
            'NRMSE_percent', 'Relative_L2_percent'});
        caseErrorTable = [caseErrorTable; newRow]; %#ok<AGROW>
    end
end
writetable(caseErrorTable, fullfile(validationDirectory, ...
    'VN_equation_validation_by_case.csv'));

disp(summaryTable);
fprintf('Independent VN validation completed in:\n%s\n', ...
    validationDirectory);
toc;
end


function saveComparisonCurve(tau, FEMValues, equationValues, ...
    coefficientName, yLimits, outputPath)
% Save one FEM-versus-equation curve comparison.

    ensureParentDirectory(outputPath);
    figureHandle = figure('Visible', 'off', 'Color', 'white', ...
        'Units', 'inches', 'Position', [0.2, 0.2, 12, 7]);
    plot(tau, FEMValues, 'LineWidth', 2.0, 'DisplayName', 'FEM');
    hold on;
    plot(tau, equationValues, '--', 'LineWidth', 2.0, ...
        'DisplayName', 'Analytical model');
    hold off;
    xlabel('T/T_v [-]');
    ylabel(sprintf('%s^* [-]', coefficientName));
    xlim([min(tau), max(tau)]);
    ylim(yLimits);
    grid on;
    box on;
    legend('Location', 'best');
    exportgraphics(figureHandle, outputPath, 'Resolution', 220);
    close(figureHandle);
end


function saveComparisonSummary(tau, FEM, prediction, ...
    coefficientDefinitions, fixedYLimits, outputPath)
% Save all eight FEM-versus-equation coefficients with fixed scales.

    ensureParentDirectory(outputPath);
    figureHandle = figure('Visible', 'off', 'Color', 'white', ...
        'Units', 'inches', 'Position', [0.2, 0.2, 18, 9]);
    tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

    for coefficient = 1:size(coefficientDefinitions, 1)
        coefficientName = coefficientDefinitions{coefficient, 1};
        FEMField = coefficientDefinitions{coefficient, 2};
        equationField = coefficientDefinitions{coefficient, 3};
        row = coefficientDefinitions{coefficient, 4};
        column = coefficientDefinitions{coefficient, 5};
        nexttile;
        FEMMatrix = FEM.(FEMField);
        equationMatrix = prediction.(equationField);
        plot(tau, squeeze(FEMMatrix(:, row, column)), ...
            'LineWidth', 2.0, 'DisplayName', 'FEM');
        hold on;
        plot(tau, squeeze(equationMatrix(:, row, column)), '--', ...
            'LineWidth', 1.8, 'DisplayName', 'Analytical model');
        hold off;
        xlabel('T/T_v [-]');
        ylabel(sprintf('%s^* [-]', coefficientName));
        title(coefficientName);
        xlim([min(tau), max(tau)]);
        ylim(fixedYLimits(coefficient, :));
        grid on;
        box on;
        if coefficient == 1
            legend('Location', 'best');
        end
    end

    exportgraphics(figureHandle, outputPath, 'Resolution', 220);
    close(figureHandle);
end
