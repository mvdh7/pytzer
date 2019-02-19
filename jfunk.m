load('testfiles/jfunk.mat')

figure(1); clf

subplot(3,1,1); hold on

    plot(I,x12)
    plot(I,x13)
    plot(I,x23)

    legend('1:2','1:3','2:3', 'location','eastoutside')

    xlabel('Ionic strength')
    ylabel('xij')

fx23 = sqrt(x23);
    
subplot(3,1,2); hold on
    
    xlim(minmax(fx23'))

    plot(fx23,J_Harvie)
    plot(fx23,J_P75_eq46)
    plot(fx23,J_P75_eq47)
    plot(fx23,J_fit)
    plot(fx23,J_fit2)
    plot(fx23,J_quad)

    legend('Harvie','P75 Eq. 46','P75 Eq. 47','refit', ...
        'refit2','quad', 'location','eastoutside')

    xlabel('sqrt(xij)')
    ylabel('J')
    
subplot(3,1,3); hold on

    xlim(minmax(fx23'))

    nl = plot(get(gca,'xlim'),[0 0],'k'); nolegend(nl)
    
    plot(fx23,J_Harvie   - J_quad)
    plot(fx23,J_P75_eq46 - J_quad)
    plot(fx23,J_P75_eq47 - J_quad)
    plot(fx23,J_fit   - J_quad)
    plot(fx23,J_fit2   - J_quad)

    legend('Harvie','P75 Eq. 46','P75 Eq. 47','refit','refit2', ...
        'location','eastoutside')

    ylim(0.5e-3*[-1 1]) % to see Harvie errors
    
    xlabel('sqrt(xij)')
    ylabel('\DeltaJ')
    