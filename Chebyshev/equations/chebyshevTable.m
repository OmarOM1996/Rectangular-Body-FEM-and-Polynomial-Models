function table = chebyshevTable(value, maximumOrder)
% Return T_0(value), ..., T_maximumOrder(value) by recurrence.

    value = value(:);
    table = zeros(numel(value), maximumOrder + 1);
    table(:, 1) = 1;
    if maximumOrder >= 1
        table(:, 2) = value;
    end
    for order = 2:maximumOrder
        table(:, order+1) = 2*value.*table(:, order) - table(:, order-1);
    end
end
