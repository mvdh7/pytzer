% Simulate dataset
rng(295)
x = (0:0.1:10)';

gr = 3;
in = 4;

yFX = @(x,gr,in) gr * x + in;

ySD = 2;

yx = yFX(x,gr,in);

y = yx + randn(size(x)) * ySD;

% Adjust gradient parameter and evaluate RMSE
rng('shuffle')

testfracG = (0.5:0.01:1.5)';
testfracI = (-0.5:0.02:2.5)';
yrms = NaN(numel(testfracG),numel(testfracI));

for G = 1:numel(testfracG)
    for I = 1:numel(testfracI)
    
        yrms(G,I) = rms(yFX(x,gr,in) + randn(size(x)) * ySD ...
            - yFX(x,gr * testfracG(G),in * testfracI(I)));
    
    end %for I
end %for G

% Plot
figure(1); clf

subplot(2,2,1); hold on

    plot([0 10],yFX([0 10],gr,in),'k', 'linewidth',1)
    scatter(x,y,30,'filled', 'markerfacealpha',0.7)
    
subplot(2,2,2); hold on

    scatter(testfracI,1./yrms(51,:).^2)
%     plot(testfracG,normpdf(testfracG,1,0.08)/20)

subplot(2,1,2); hold on

    contourf(testfracI,testfracG,1./yrms,64, 'edgecolor','none')
    colormap(magma(64))
    
    xlabel('Intercept')
    ylabel('Gradient')
    
    axis equal
    