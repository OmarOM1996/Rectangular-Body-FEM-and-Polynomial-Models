function basis = chebyshevBasis(x, y, z, termOrders)
% Evaluate selected three-variable Chebyshev-product terms.

    maximumOrders = max(termOrders, [], 1);
    Tx = chebyshevTable(x, maximumOrders(1));
    Ty = chebyshevTable(y, maximumOrders(2));
    Tz = chebyshevTable(z, maximumOrders(3));

    basis = zeros(numel(x), size(termOrders, 1));
    for term = 1:size(termOrders, 1)
        orders = termOrders(term, :);
        basis(:, term) = Tx(:, orders(1)+1) .* ...
            Ty(:, orders(2)+1) .* Tz(:, orders(3)+1);
    end
end
