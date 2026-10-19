function termOrders = hydrodynamicTermOrders()
% Generate the Chebyshev terms for the hydrodynamic equations.
%
% Columns:
% 1: order in T/Tv
% 2: order in a/h
% 3: order in d/a

maximumOrders = [12, 3, 3];

termOrders = zeros(0, 3);

for tauOrder = 0:maximumOrders(1)
    for aOverHOrder = 0:maximumOrders(2)
        for dOverAOrder = 0:maximumOrders(3)
            normalizedOrder = tauOrder/maximumOrders(1) + ...
                aOverHOrder/maximumOrders(2) + ...
                dOverAOrder/maximumOrders(3);
            if normalizedOrder <= 1.55 && ...
                    aOverHOrder+dOverAOrder <= 4
                termOrders(end+1, :) = ...
                    [tauOrder, aOverHOrder, dOverAOrder]; %#ok<AGROW>
            end
        end
    end
end
end
