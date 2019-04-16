c = [1.5 4.5];
a = [1.0 2.0 3.0];
a1 = 2.5;
I = 0.5 * (sum(c) + sum(a));
rng(7);
B0ca = randn(numel(c), numel(a));
B1ca = randn(numel(c), numel(a));
g = @(x) 2 * (1 - (1 + x) * exp(-x)) / x^2;
b0lp = 0;
for C = 1:numel(c)
    for A = 1:numel(a)
        b0lp = b0lp + c(C)*a(A)*(B0ca(C, A) + B1ca(C, A)*g(a1*sqrt(I)));
    end % for A
end % for C
b0mx1 = c * B0ca * a' + c * B1ca * a' .* g(a1*sqrt(I));
b0mx2 = c * (B0ca + B1ca*g(a1*sqrt(I))) * a';
