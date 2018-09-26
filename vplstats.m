load('pickles/simpar_vpl.mat')
vplbase = readtable('pickles/simpar_vpl.csv');
vplsrcs.all.srcs = unique(vplbase.src);

eles = {'KCl' 'NaCl'};

% Define marker styles
mrks = repmat({'o' 'v' '^' '<' '>' 'sq' 'd' 'p' 'h'},1,3);
msms = repmat([ 1   1   1   1   1   1    1   3   1 ],1,3);
clrs = repmat([228,26,28; 55,126,184; 77,175,74; 152,78,163; 255,127,0; 
    166,86,40; 247,129,191; 153,153,153] / 255,3,1);
for S = 1:numel(vplsrcs.all.srcs)
    fmrk.(vplsrcs.all.srcs{S}) = mrks{S};
    fclr.(vplsrcs.all.srcs{S}) = clrs(S,:);
    fmsm.(vplsrcs.all.srcs{S}) = msms(S);
end %for S
mksz = 50;

figure(1); clf; hold on

% Plot data by source
for E = 1:numel(eles)
ele = eles{E};
EL = strcmp(vplbase.ele,ele);
vplsrcs.(ele).srcs = unique(vplbase.src(EL));
for S = 1:numel(vplsrcs.(ele).srcs)

    src = vplsrcs.(ele).srcs{S};
    SL = EL & strcmp(vplbase.src,src);
    SL = SL & vplbase.t == 298.15;

    Sy = [];
    
    if any(SL)
        Sx = linspace(min(vplbase.m(SL)),max(vplbase.m(SL)),100);
        Sy = vplerr_rdm.(ele).(src)(2) .* exp(-Sx) ...
            + vplerr_rdm.(ele).(src)(1);
        
        scatter(mean(Sy), ...mean(Sy), ...
            abs(vplerr_sys.(ele).(src)(1)), ...
            mksz*fmsm.(src),fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)
    end %if
    
    
    
%     if any(SL)
%         flegs{end+1} = src;
%     end %if

end %for S
end %for E