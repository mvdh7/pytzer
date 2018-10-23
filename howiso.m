true_a = 43;
true_b = 56;

sd_a = 1;
sd_b = 3;

reps = 1e6;

meas_a = true_a + randn(reps,1) * sd_a;
meas_b = true_b + randn(reps,1) * sd_b;

clc
disp(['a = ' num2str(mean(meas_a)) ' ' plusminus ' ' num2str(std(meas_a))])
disp(['b = ' num2str(mean(meas_b)) ' ' plusminus ' ' num2str(std(meas_b))])

% Add sum
true_r = true_b / true_a;
sd_r = 0.1;

% Direct ratio measurements
meas_r = true_r + randn(reps,1) * sd_r;

% From a and b measurements
meas_r_ab = meas_b ./ meas_a;
sdx_meas_r_ab = true_r * sqrt((sd_b/true_b)^2 + (sd_a/true_a)^2);

% etc.

% Best estimates
meas_ar = meas_a .* meas_r;
sd_meas_ar = std(meas_ar);
sd_ar = true_b * sqrt((sd_a/true_a)^2 + (sd_r/true_r)^2);

best_b = (meas_b / sd_b^2 + meas_ar / sd_ar^2) ...
    / (1/sd_b^2 + 1/sd_meas_ar^2);

sd_best_b_meas = std(best_b);
sd_best_b_calc = sqrt((1/sd_ar^2 + 1/sd_b^2) ...
    * (sd_ar^2*sd_b^2/(sd_ar^2 + sd_b^2))^2);

meas_br = meas_b ./ meas_r;
sd_meas_br = std(meas_br);
sd_br = true_a * sqrt((sd_b/true_b)^2 + (sd_r/true_r)^2);

best_a = (meas_a / sd_a^2 + meas_br / sd_br^2) ...
     / (1/sd_a^2 + 1/sd_meas_br^2);

sd_best_a_meas = std(best_a);
sd_best_a_calc = sqrt((1/sd_br^2 + 1/sd_a^2) ...
    * (sd_br^2*sd_a^2/(sd_br^2 + sd_a^2))^2);
 
cov_ab = cov([meas_a meas_b]);
cov_best_ab = cov([best_a best_b]);

% Plot simulation results
figure(1); clf

subplot(2,2,1); hold on

    histogram(meas_a, 'edgecolor','none', 'facecolor','b', ...
        'normalization','pdf')
    histogram(meas_b, 'edgecolor','none', 'facecolor','r', ...
        'normalization','pdf')
    
    histogram(meas_ar, 'edgecolor','none', 'facecolor','g', ...
        'normalization','pdf')
    histogram(best_b, 'edgecolor','none', 'facecolor','m', ...
        'normalization','pdf')
    
    histogram(meas_br, 'edgecolor','none', 'facecolor','g', ...
        'normalization','pdf')
    histogram(best_a, 'edgecolor','none', 'facecolor','m', ...
        'normalization','pdf')

    fx = 8:0.1:32;
    
    plot(fx,normpdf(fx,true_a,sd_a),'b')
    plot(fx,normpdf(fx,true_b,sd_b),'r')
    
    legend('a','b','ar','best b')

subplot(2,2,2); hold on

    histogram(meas_r,    'edgecolor','none', 'facecolor','k', ...
        'normalization','pdf')
    histogram(meas_r_ab, 'edgecolor','none', 'facecolor','m', ...
        'normalization','pdf')
    
    fx = 1.4:0.001:3;
    
    plot(fx,normpdf(fx,true_r,sd_r),'k')
    plot(fx,normpdf(fx,true_r,sdx_meas_r_ab),'m')
    
    legend('r direct','r from b/a')

test_a = meas_a + meas_br;
test_b = meas_b + meas_ar;
    
cv_testa_measr = cov([test_a meas_r]);
disp(cv_testa_measr)
disp(sd_r^2*-true_b/true_r^2)

% cv_tests = cov([test_a test_b])
% cv_tests_calc = 0 + true_r*sd_a^2 + sd_b^2/true_r
    
%% See covariance matrices
covmx_viz(2,[true_a true_b],cov_ab,{'a' 'b'},'');
covmx_viz(3,[true_a true_b],cov([best_b meas_r]),{'a' 'b'},'');
