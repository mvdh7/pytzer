D = randn(1e5,1);
maD = mean(abs(D));

guess_SD = abs(D) * sqrt(pi/2);

S = randn(1e5,1) * maD;

maS = mean(abs(S));
% 
% figure(1); clf; hold on
% 
% histogram(abs(D),0:0.05:4.5)
% histogram(abs(S),0:0.05:4.5)
% 
% xlim([0 4.5])
% ylim([0 6500])
% 
% plot(maD*[1 1],get(gca,'ylim'),'b', 'linewidth',3)
% plot(maS*[1 1],get(gca,'ylim'),'r', 'linewidth',3)
% plot(std(S)*[1 1],get(gca,'ylim'),'r--', 'linewidth',3)
% 
% %%
% figure(2); clf; hold on
% 
% histogram(guess_SD)
% 
% %%
figure(3); clf

subplot(2,1,1)
scatter(1:1e5,D,10,'filled', 'markerfacealpha',0.1)
ylim(5*[-1 1])

subplot(2,1,2); hold on
scatter(1:1e5,D,10,'filled', 'markerfacealpha',0.1)
scatter(1:1e5,S,10,'filled', 'markerfacealpha',0.1)
ylim(5*[-1 1])