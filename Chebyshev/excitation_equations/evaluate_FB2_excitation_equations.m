function Fstar = evaluate_FB2_excitation_equations(tau, aOverH, dOverA)
% Evaluate the six dimensionless complex-excitation equations.
% Output columns are F1*, F3*, and F5*.

equationDirectory = fileparts(mfilename('fullpath'));
projectDirectory = fileparts(equationDirectory);
addpath(fullfile(projectDirectory, 'equations'));

model = load(fullfile(equationDirectory, ...
    'FB2_excitation_equation_coefficients.mat'));

tau = tau(:);
if isscalar(aOverH)
    aOverH = repmat(aOverH, size(tau));
else
    aOverH = aOverH(:);
end
if isscalar(dOverA)
    dOverA = repmat(dOverA, size(tau));
else
    dOverA = dOverA(:);
end
if ~isequal(size(tau), size(aOverH), size(dOverA))
    error('tau, aOverH and dOverA must be scalars or equal-sized vectors.');
end

x = scaleVariable(tau, model.tau_range);
y = scaleVariable(aOverH, model.a_over_h_range);
z = scaleVariable(dOverA, model.d_over_a_range);
basis = chebyshevBasis(x, y, z, model.term_orders);
components = basis*model.coefficient_matrix;

Fstar = complex(zeros(numel(tau), 3));
Fstar(:, 1) = complex(components(:, 1), components(:, 2));
Fstar(:, 2) = complex(components(:, 3), components(:, 4));
Fstar(:, 3) = complex(components(:, 5), components(:, 6));
end

function scaled = scaleVariable(value, range)
scaled = 2*(value-range(1))/(range(2)-range(1))-1;
end
