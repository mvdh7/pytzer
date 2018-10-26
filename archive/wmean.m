% https://en.wikipedia.org/wiki/Weighted_arithmetic_mean

%% Vector-valued estimates
x1 = [1;0];
x2 = [0;1];

m1 = [1 0; 0 100];
m2 = [100 0; 0 1];

x = inv(inv(m1) + inv(m2)) * (inv(m1)*x1 + inv(m2)*x2);

%% Accounting for correlations
X = [10;20];
M = [4 1;
     1 3];
J = [1;1];
 
S = inv(J' * inv(M) * J);
xx = S * (J' * inv(M) * X);
