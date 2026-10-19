function [addedMass, damping, Astar, Bstar] = ...
    evaluate_FB2_dimensional_coefficients(tau, a, d, h, rho, gravity)
% Evaluate dimensional and nondimensional FB2 hydrodynamic coefficients.

    validateattributes(a, {'numeric'}, {'real', 'finite', 'positive', 'scalar'});
    validateattributes(d, {'numeric'}, {'real', 'finite', 'positive', 'scalar'});
    validateattributes(h, {'numeric'}, {'real', 'finite', 'positive', 'scalar'});
    validateattributes(rho, {'numeric'}, {'real', 'finite', 'positive', 'scalar'});
    validateattributes(gravity, {'numeric'}, {'real', 'finite', 'positive', 'scalar'});

    [Astar, Bstar] = evaluate_FB2_hydrodynamic_equations(tau, a/h, d/a);

    translationScale = 2*rho*a*d;
    pitchScale = 8*rho*a^3*d;
    modeScales = [translationScale, translationScale, pitchScale];
    matrixScale = sqrt(modeScales(:) * modeScales(:).');

    addedMass = zeros(size(Astar));
    damping = zeros(size(Bstar));
    for point = 1:numel(tau)
        addedMass(point, :, :) = squeeze(Astar(point, :, :)) .* matrixScale;
        damping(point, :, :) = squeeze(Bstar(point, :, :)) .* ...
            (sqrt(gravity/a) * matrixScale);
    end
end
