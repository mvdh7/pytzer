[num,txt] = xlsread('dis.xlsx','DIS data');

m = num(:,2);
a = num(:,3);
t = num(:,1);
s = txt(4:end,1);

clear num txt

figure(1); clf; hold on

us = unique(s);

marks = {'o' 'sq' 'v' '^' 'd' 'x' '+'};

for S = 1:numel(us)

    SL = strcmp(s,us{S});
    scatter(sqrt(m(SL)),a(SL),50,t(SL), 'marker',marks{S}, 'linewidth',2)
    
end %for S

legend(us)
colorbar
grid on

clear S SL
