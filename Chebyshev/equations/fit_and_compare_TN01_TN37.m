function fitSummary = fit_and_compare_TN01_TN37(projectDirectory, ...
    manualYLimits, createPerCaseComparisons, createOverviewComparisons, ...
    caseFilter)
% Fit the hydrodynamic equations and compare them with TN01--TN37 FEM data.
% TN01--TN37 form the fitting dataset. These plots are therefore called
% FEM--equation comparisons, not independent validation figures.

if nargin < 2
    manualYLimits = [];
end
if nargin < 3
    createPerCaseComparisons = true;
end
if nargin < 4
    createOverviewComparisons = true;
end
if nargin < 5
    caseFilter = "ALL";
end

equationDirectory = fullfile(projectDirectory, 'equations');
dataDirectory = fullfile(projectDirectory, 'data_mat');
comparisonRoot = fullfile(equationDirectory, ...
    'TN_FEM_equation_comparisons');
overviewDirectory = fullfile(comparisonRoot, 'comparison_overviews');
if ~isfolder(overviewDirectory)
    mkdir(overviewDirectory);
end

addpath(projectDirectory);
addpath(equationDirectory);

% Generate the common 109-term Chebyshev basis.
commonTermOrders = hydrodynamicTermOrders();
linearTermOrders = commonTermOrders;
B11TermOrders = commonTermOrders;
B33TermOrders = commonTermOrders;
B55TermOrders = commonTermOrders;

fprintf('Number of hydrodynamic terms: %d\n', ...
    size(commonTermOrders, 1));

coefficientNames = {'A11', 'A33', 'A55', 'A15', ...
    'B11', 'B33', 'B55', 'B15'};
linearCoefficientNames = {'A11', 'A33', 'A55', 'A15', 'B15'};
numberOfCases = 37;
numberOfCoefficients = numel(coefficientNames);

resultFiles = dir(fullfile(dataDirectory, 'TN*_results.mat'));
if numel(resultFiles) ~= numberOfCases
    error('Expected 37 TN MAT files, found %d.', numel(resultFiles));
end

caseNumbers = zeros(numberOfCases, 1);
for fileNumber = 1:numel(resultFiles)
    token = regexp(resultFiles(fileNumber).name, '^TN(\d+)', ...
        'tokens', 'once');
    caseNumbers(fileNumber) = str2double(token{1});
end
[~, order] = sort(caseNumbers);
resultFiles = resultFiles(order);

firstResult = load(fullfile(resultFiles(1).folder, resultFiles(1).name));
tau = firstResult.T_over_Tv(:);
numberOfTauPoints = numel(tau);
FEMCurves = zeros(numberOfCases, numberOfTauPoints, numberOfCoefficients);
fingerprints = strings(numberOfCases, 1);
geometries = zeros(numberOfCases, 2);

for caseNumber = 1:numberOfCases
    result = load(fullfile(resultFiles(caseNumber).folder, ...
        resultFiles(caseNumber).name));
    fingerprint = strtrim(string(result.fingerprint));
    fingerprints(caseNumber) = fingerprint(1);
    caseTau = result.T_over_Tv(:);
    if max(abs(caseTau-tau)) > 1.0e-12
        error('%s uses different T/Tv points.', fingerprints(caseNumber));
    end

    a = double(result.a);
    d = double(result.d);
    h = double(result.h);
    geometries(caseNumber, :) = [a/h, d/a];
    Astar = result.added_mass_nondimensional;
    Bstar = result.radiation_damping_nondimensional;

    FEMCurves(caseNumber, :, 1) = squeeze(Astar(:, 1, 1));
    FEMCurves(caseNumber, :, 2) = squeeze(Astar(:, 2, 2));
    FEMCurves(caseNumber, :, 3) = squeeze(Astar(:, 3, 3));
    FEMCurves(caseNumber, :, 4) = squeeze(Astar(:, 1, 3));
    FEMCurves(caseNumber, :, 5) = squeeze(Bstar(:, 1, 1));
    FEMCurves(caseNumber, :, 6) = squeeze(Bstar(:, 2, 2));
    FEMCurves(caseNumber, :, 7) = squeeze(Bstar(:, 3, 3));
    FEMCurves(caseNumber, :, 8) = squeeze(Bstar(:, 1, 3));
end

tauRange = [min(tau), max(tau)];
aOverHRange = [min(geometries(:, 1)), max(geometries(:, 1))];
dOverARange = [min(geometries(:, 2)), max(geometries(:, 2))];

tauAll = repmat(tau, numberOfCases, 1);
aOverHAll = repelem(geometries(:, 1), numberOfTauPoints);
dOverAAll = repelem(geometries(:, 2), numberOfTauPoints);
x = scaleToChebyshev(tauAll, tauRange);
y = scaleToChebyshev(aOverHAll, aOverHRange);
z = scaleToChebyshev(dOverAAll, dOverARange);

targets = reshape(permute(FEMCurves, [2, 1, 3]), ...
    [], numberOfCoefficients);
linearTargetColumns = [1, 2, 3, 4, 8];
linearBasis = chebyshevBasis(x, y, z, linearTermOrders);
linearCoefficientMatrix = linearBasis \ targets(:, linearTargetColumns);

B11Basis = chebyshevBasis(x, y, z, B11TermOrders);
B33Basis = chebyshevBasis(x, y, z, B33TermOrders);
B55Basis = chebyshevBasis(x, y, z, B55TermOrders);
B11SqrtCoefficients = B11Basis \ sqrt(max(targets(:, 5), 0));
B33SqrtCoefficients = B33Basis \ sqrt(max(targets(:, 6), 0));
B55SqrtCoefficients = B55Basis \ sqrt(max(targets(:, 7), 0));

predictedTargets = zeros(size(targets));
linearPrediction = linearBasis * linearCoefficientMatrix;
predictedTargets(:, linearTargetColumns) = linearPrediction;
predictedTargets(:, 5) = (B11Basis * B11SqrtCoefficients).^2;
predictedTargets(:, 6) = (B33Basis * B33SqrtCoefficients).^2;
predictedTargets(:, 7) = (B55Basis * B55SqrtCoefficients).^2;
equationCurves = permute(reshape(predictedTargets, ...
    numberOfTauPoints, numberOfCases, numberOfCoefficients), [2, 1, 3]);

trainingNRMSE = zeros(numberOfCoefficients, 1);
caseNRMSE = zeros(numberOfCases, numberOfCoefficients);
caseRelativeL2 = zeros(numberOfCases, numberOfCoefficients);
fixedYLimits = zeros(numberOfCoefficients, 2);

for coefficient = 1:numberOfCoefficients
    FEMValues = targets(:, coefficient);
    equationValues = predictedTargets(:, coefficient);
    coefficientRange = max(FEMValues)-min(FEMValues);
    trainingNRMSE(coefficient) = ...
        sqrt(mean((FEMValues-equationValues).^2))/max(coefficientRange, eps);

    combinedMinimum = min([FEMValues; equationValues]);
    combinedMaximum = max([FEMValues; equationValues]);
    padding = 0.05*max(combinedMaximum-combinedMinimum, eps);
    fixedYLimits(coefficient, :) = ...
        [combinedMinimum-padding, combinedMaximum+padding];

    for caseNumber = 1:numberOfCases
        caseFEM = squeeze(FEMCurves(caseNumber, :, coefficient)).';
        caseEquation = squeeze(equationCurves(caseNumber, :, coefficient)).';
        caseError = caseFEM-caseEquation;
        caseRange = max(caseFEM)-min(caseFEM);
        caseNRMSE(caseNumber, coefficient) = ...
            sqrt(mean(caseError.^2))/max(caseRange, eps);
        caseRelativeL2(caseNumber, coefficient) = ...
            norm(caseError)/max(norm(caseFEM), eps);
    end
end

if ~isempty(manualYLimits)
    if ~isequal(size(manualYLimits), [numberOfCoefficients, 2]) || ...
            any(manualYLimits(:, 1) >= manualYLimits(:, 2))
        error(['COMPARISON_Y_LIMITS must be empty or an 8-by-2 matrix ', ...
            'with increasing limits.']);
    end
    fixedYLimits = manualYLimits;
end

% Save the active 37-case equation model used by the evaluator.
coefficient_names = coefficientNames;  
linear_coefficient_names = linearCoefficientNames;  
tau_range = tauRange;  
a_over_h_range = aOverHRange;  
d_over_a_range = dOverARange;  
linear_term_orders = linearTermOrders;  
linear_coefficient_matrix = linearCoefficientMatrix;  
training_nrmse = trainingNRMSE;  
B11_term_orders = B11TermOrders;  
B11_sqrt_coefficients = B11SqrtCoefficients;  
B33_term_orders = B33TermOrders;  
B33_sqrt_coefficients = B33SqrtCoefficients;  
B55_term_orders = B55TermOrders;  
B55_sqrt_coefficients = B55SqrtCoefficients;  
T_over_Tv = tau;  
FEM_curves = FEMCurves;  
final_equation_prediction = equationCurves;  
model_version = 'TN01--TN37 Atlas 109-term fit';  

save(fullfile(equationDirectory, ...
    'FB2_preliminary_equation_coefficients.mat'), ...
    'coefficient_names', 'linear_coefficient_names', ...
    'tau_range', 'a_over_h_range', 'd_over_a_range', ...
    'linear_term_orders', 'linear_coefficient_matrix', ...
    'training_nrmse', 'fingerprints', 'geometries', 'T_over_Tv', ...
    'FEM_curves', 'final_equation_prediction', ...
    'B11_term_orders', 'B11_sqrt_coefficients', ...
    'B33_term_orders', 'B33_sqrt_coefficients', ...
    'B55_term_orders', 'B55_sqrt_coefficients', 'model_version');

writeCoefficientTables(equationDirectory, linearTermOrders, ...
    linearCoefficientMatrix, linearCoefficientNames, ...
    B11TermOrders, B11SqrtCoefficients, ...
    B33TermOrders, B33SqrtCoefficients, ...
    B55TermOrders, B55SqrtCoefficients);

% Eight fixed-scale overview figures, each containing TN01--TN37.
if createOverviewComparisons
for coefficient = 1:numberOfCoefficients
    figureHandle = figure('Visible', 'off', 'Color', 'white', ...
        'Units', 'inches', 'Position', [0.2, 0.2, 24, 12]);
    tiledlayout(5, 8, 'TileSpacing', 'compact', 'Padding', 'compact');
    for caseNumber = 1:numberOfCases
        nexttile;
        plot(tau, squeeze(FEMCurves(caseNumber, :, coefficient)), ...
            'LineWidth', 1.6, 'DisplayName', 'FEM');
        hold on;
        plot(tau, squeeze(equationCurves(caseNumber, :, coefficient)), ...
            '--', 'LineWidth', 1.5, 'DisplayName', 'Analytical model');
        hold off;
        xlim(tauRange);
        ylim(fixedYLimits(coefficient, :));
        title(sprintf('a/h=%.2f, d/a=%.2f', ...
            geometries(caseNumber, 1), geometries(caseNumber, 2)), ...
            'FontSize', 8);
        grid on;
        box on;
        if caseNumber > 32
            xlabel('T/T_v [-]');
        end
        if mod(caseNumber-1, 8) == 0
            ylabel(sprintf('%s^* [-]', coefficientNames{coefficient}));
        end
        if caseNumber == 1
            legend('Location', 'best', 'FontSize', 7);
        end
    end
    exportgraphics(figureHandle, fullfile(overviewDirectory, sprintf( ...
        '%s_FEM_vs_equation_TN01_to_TN37.png', ...
        coefficientNames{coefficient})), 'Resolution', 220);
    close(figureHandle);
end
end

% A separate fixed-scale comparison set for every TN case.
if createPerCaseComparisons
for caseNumber = 1:numberOfCases
    if caseFilter ~= "ALL" && ...
            ~startsWith(fingerprints(caseNumber), caseFilter + "_", ...
            'IgnoreCase', true)
        continue;
    end
    caseDirectory = fullfile(comparisonRoot, char(fingerprints(caseNumber)));
    for coefficient = 1:numberOfCoefficients
        outputPath = fullfile(caseDirectory, sprintf( ...
            '%s_FEM_vs_equation_%s.png', fingerprints(caseNumber), ...
            coefficientNames{coefficient}));
        saveFixedComparison(tau, ...
            squeeze(FEMCurves(caseNumber, :, coefficient)).', ...
            squeeze(equationCurves(caseNumber, :, coefficient)).', ...
            coefficientNames{coefficient}, fixedYLimits(coefficient, :), ...
            tauRange, outputPath);
    end
    saveCaseSummary(tau, squeeze(FEMCurves(caseNumber, :, :)), ...
        squeeze(equationCurves(caseNumber, :, :)), coefficientNames, ...
        fixedYLimits, tauRange, fullfile(caseDirectory, sprintf( ...
        '%s_FEM_vs_equation_summary.png', fingerprints(caseNumber))));
    fprintf('%s: FEM--equation comparisons complete\n', fingerprints(caseNumber));
end
end

coefficientColumn = string(coefficientNames(:));
worstCaseNRMSE = max(caseNRMSE, [], 1).';
meanRelativeL2 = mean(caseRelativeL2, 1).';
fitSummary = table(coefficientColumn, 100*trainingNRMSE, ...
    100*worstCaseNRMSE, 100*meanRelativeL2, ...
    'VariableNames', {'Coefficient', 'Global_NRMSE_percent', ...
    'Worst_case_NRMSE_percent', 'Mean_relative_L2_percent'});
writetable(fitSummary, fullfile(comparisonRoot, ...
    'TN01_TN37_equation_fit_summary.csv'));

axisLimitsTable = table(coefficientColumn, fixedYLimits(:, 1), ...
    fixedYLimits(:, 2), 'VariableNames', ...
    {'Coefficient', 'Fixed_y_min', 'Fixed_y_max'});
writetable(axisLimitsTable, fullfile(comparisonRoot, ...
    'TN01_TN37_fixed_comparison_y_limits.csv'));

caseTable = table;
for caseNumber = 1:numberOfCases
    for coefficient = 1:numberOfCoefficients
        row = table(fingerprints(caseNumber), ...
            string(coefficientNames{coefficient}), ...
            100*caseNRMSE(caseNumber, coefficient), ...
            100*caseRelativeL2(caseNumber, coefficient), ...
            'VariableNames', {'Fingerprint', 'Coefficient', ...
            'NRMSE_percent', 'Relative_L2_percent'});
        caseTable = [caseTable; row]; %#ok<AGROW>
    end
end
writetable(caseTable, fullfile(comparisonRoot, ...
    'TN01_TN37_equation_fit_by_case.csv'));

writeEquationReport(equationDirectory, tauRange, aOverHRange, ...
    dOverARange, coefficientNames, trainingNRMSE);

disp(fitSummary);
fprintf('TN01--TN37 FEM--equation comparisons created in:\n%s\n', ...
    comparisonRoot);
end


function scaled = scaleToChebyshev(values, limits)
    scaled = 2*(values-limits(1))/diff(limits)-1;
end


function saveFixedComparison(tau, FEMValues, equationValues, ...
    coefficientName, yLimits, xLimits, outputPath)
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
    xlim(xLimits);
    ylim(yLimits);
    grid on;
    box on;
    legend('Location', 'best');
    exportgraphics(figureHandle, outputPath, 'Resolution', 220);
    close(figureHandle);
end


function saveCaseSummary(tau, FEMValues, equationValues, ...
    coefficientNames, fixedYLimits, xLimits, outputPath)
    ensureParentDirectory(outputPath);
    figureHandle = figure('Visible', 'off', 'Color', 'white', ...
        'Units', 'inches', 'Position', [0.2, 0.2, 18, 9]);
    tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    for coefficient = 1:numel(coefficientNames)
        nexttile;
        plot(tau, FEMValues(:, coefficient), 'LineWidth', 1.8, ...
            'DisplayName', 'FEM');
        hold on;
        plot(tau, equationValues(:, coefficient), '--', ...
            'LineWidth', 1.7, 'DisplayName', 'Analytical model');
        hold off;
        xlabel('T/T_v [-]');
        ylabel(sprintf('%s^* [-]', coefficientNames{coefficient}));
        title(coefficientNames{coefficient});
        xlim(xLimits);
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


function writeCoefficientTables(equationDirectory, linearOrders, ...
    linearCoefficients, linearNames, B11Orders, B11Coefficients, ...
    B33Orders, B33Coefficients, B55Orders, B55Coefficients)
    orderNames = {'tau_order', 'p_order', 'q_order'};
    linearTable = array2table([linearOrders, linearCoefficients], ...
        'VariableNames', [orderNames, linearNames]);
    writetable(linearTable, fullfile(equationDirectory, ...
        'FB2_linear_equation_coefficients.csv'));
    writeSqrtTable(equationDirectory, 'B11', B11Orders, B11Coefficients);
    writeSqrtTable(equationDirectory, 'B33', B33Orders, B33Coefficients);
    writeSqrtTable(equationDirectory, 'B55', B55Orders, B55Coefficients);
end


function writeSqrtTable(equationDirectory, name, orders, coefficients)
    coefficientTable = array2table([orders, coefficients(:)], ...
        'VariableNames', {'tau_order', 'p_order', 'q_order', ...
        'sqrt_coefficient'});
    writetable(coefficientTable, fullfile(equationDirectory, sprintf( ...
        'FB2_%s_sqrt_equation_coefficients.csv', name)));
end


function writeEquationReport(equationDirectory, tauRange, ...
    aOverHRange, dOverARange, coefficientNames, trainingNRMSE)
    path = fullfile(equationDirectory, ...
        'FB2_preliminary_equation_report.txt');
    fileID = fopen(path, 'w');
    cleanup = onCleanup(@() fclose(fileID));  
    fprintf(fileID, 'FB2 extended explicit hydrodynamic correlations\n\n');
    fprintf(fileID, 'Fitting dataset: TN01--TN37.\n');
    fprintf(fileID, 'Variables: tau=T/Tv, p=a/h, q=d/a.\n');
    fprintf(fileID, 'Validity range: %.2f <= tau <= %.2f, ', tauRange);
    fprintf(fileID, '%.2f <= a/h <= %.2f, ', aOverHRange);
    fprintf(fileID, '%.2f <= d/a <= %.2f.\n\n', dOverARange);
    fprintf(fileID, ['A11*, A33*, A55*, A15* and B15* use one ', ...
        'three-variable Chebyshev expansion.\n']);
    fprintf(fileID, ['B11*, B33* and B55* use the square of a ', ...
        'Chebyshev expansion to prevent negative damping.\n']);
    fprintf(fileID, 'A51*=A15* and B51*=B15*.\n\n');
    fprintf(fileID, 'Fit errors (global NRMSE):\n');
    for coefficient = 1:numel(coefficientNames)
        fprintf(fileID, '%s*: %.6f%%\n', coefficientNames{coefficient}, ...
            100*trainingNRMSE(coefficient));
    end
end
