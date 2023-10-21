function [x, tau] = solve_L0p(y, lambda, p)

    % solve (x-y)^2 + lambda * (|x|^0+|x|^p)
    iter_N = 5;
    xp = (lambda * (1-p))^(1/(2-p));
    for iter = 1 : iter_N
        xp = sqrt(lambda + lambda * (1-p) * xp^p);
    end
    tau = xp + 0.5 * lambda * p * xp^(p-1);
    x = zeros(size(y));
    index = find(abs(y)>tau);
    if length(index) >= 1
        y_ = y(index);
        s = abs(y_);
        for iter = 1 : iter_N
           s = abs(y_) - 0.5*lambda * p *s .^(p-1);
        end
        x(index)=sign(y_) .* s;
    end
end
