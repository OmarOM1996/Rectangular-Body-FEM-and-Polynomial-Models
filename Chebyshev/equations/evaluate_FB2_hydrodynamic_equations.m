function [Astar, Bstar] = evaluate_FB2_hydrodynamic_equations(tau, aOverH, dOverA)
% Evaluate the preliminary explicit FB2 hydrodynamic correlations.
%
% Inputs
%   tau     T/Tv, scalar or vector
%   aOverH  a/h, scalar
%   dOverA  d/a, scalar
%
% Outputs
%   Astar   nondimensional added-mass matrices, size [numel(tau),3,3]
%   Bstar   nondimensional damping matrices, size [numel(tau),3,3]
%
% Matrix rows and columns correspond to surge, heave and pitch (1,3,5).

    persistent equationData
    if isempty(equationData)
        equationDirectory = fileparts(mfilename('fullpath'));
        equationData = load(fullfile(equationDirectory, ...
            'FB2_preliminary_equation_coefficients.mat'));
    end

    tau = tau(:);
    validateattributes(tau, {'numeric'}, {'real', 'finite', 'nonempty'});
    validateattributes(aOverH, {'numeric'}, {'real', 'finite', 'scalar'});
    validateattributes(dOverA, {'numeric'}, {'real', 'finite', 'scalar'});

    if any(tau < equationData.tau_range(1) | tau > equationData.tau_range(2))
        error('tau must be between %.2f and %.2f.', ...
            equationData.tau_range(1), equationData.tau_range(2));
    end
    if aOverH < equationData.a_over_h_range(1) || ...
            aOverH > equationData.a_over_h_range(2)
        error('a/h must be between %.2f and %.2f.', ...
            equationData.a_over_h_range(1), equationData.a_over_h_range(2));
    end
    if dOverA < equationData.d_over_a_range(1) || ...
            dOverA > equationData.d_over_a_range(2)
        error('d/a must be between %.2f and %.2f.', ...
            equationData.d_over_a_range(1), equationData.d_over_a_range(2));
    end

    x = 2*(tau - equationData.tau_range(1)) / diff(equationData.tau_range) - 1;
    y = 2*(aOverH - equationData.a_over_h_range(1)) / ...
        diff(equationData.a_over_h_range) - 1;
    z = 2*(dOverA - equationData.d_over_a_range(1)) / ...
        diff(equationData.d_over_a_range) - 1;

    linearBasis = chebyshevBasis(x, y, z, equationData.linear_term_orders);
    linearValues = linearBasis * equationData.linear_coefficient_matrix;
    B11 = (chebyshevBasis(x, y, z, equationData.B11_term_orders) * ...
        equationData.B11_sqrt_coefficients(:)).^2;
    B33 = (chebyshevBasis(x, y, z, equationData.B33_term_orders) * ...
        equationData.B33_sqrt_coefficients(:)).^2;
    B55 = (chebyshevBasis(x, y, z, equationData.B55_term_orders) * ...
        equationData.B55_sqrt_coefficients(:)).^2;
    Astar = zeros(numel(tau), 3, 3);
    Bstar = zeros(numel(tau), 3, 3);

    Astar(:, 1, 1) = linearValues(:, 1);
    Astar(:, 2, 2) = linearValues(:, 2);
    Astar(:, 3, 3) = linearValues(:, 3);
    Astar(:, 1, 3) = linearValues(:, 4);
    Astar(:, 3, 1) = linearValues(:, 4);

    Bstar(:, 1, 1) = B11;
    Bstar(:, 2, 2) = B33;
    Bstar(:, 3, 3) = B55;
    Bstar(:, 1, 3) = linearValues(:, 5);
    Bstar(:, 3, 1) = linearValues(:, 5);
end
