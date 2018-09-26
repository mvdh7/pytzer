load('pickles/simpar_fpd.mat')
fpdbase = readtable('pickles/simpar_fpd.csv');
fpdsrcs.all.srcs = unique(fpdbase.src);

eles = {'KCl' 'NaCl' 'CaCl2'};

% Define marker styles
mrks = repmat({'o' 'v' '^' '<' '>' 'sq' 'd' 'p' 'h'},1,3);
msms = repmat([ 1   1   1   1   1   1    1   3   1 ],1,3);
clrs = repmat([228,26,28; 55,126,184; 77,175,74; 152,78,163; 255,127,0; 
    166,86,40; 247,129,191; 153,153,153] / 255,3,1);
for S = 1:numel(fpdsrcs.all.srcs)
    fmrk.(fpdsrcs.all.srcs{S}) = mrks{S};
    fclr.(fpdsrcs.all.srcs{S}) = clrs(S,:);
    fmsm.(fpdsrcs.all.srcs{S}) = msms(S);
end %for S
mksz = 50;

figure(2); clf; hold on

% Plot data by source
for E = 1:numel(eles)
ele = eles{E};
EL = strcmp(fpdbase.ele,ele);
fpdsrcs.(ele).srcs = unique(fpdbase.src(EL));
for S = 1:numel(fpdsrcs.(ele).srcs)


    src = fpdsrcs.(ele).srcs{S};
    SL = EL & strcmp(fpdbase.src,src);
    
    Sy = [];
    
    if any(SL)
        Sx = linspace(min(fpdbase.m(SL)),max(fpdbase.m(SL)),100);
        Sy = fpderr_rdm.(ele).(src)(2) .* Sx ...
            + fpderr_rdm.(ele).(src)(1);
        
        scatter(mean(Sy), ...sqrt(sum(fpderr_rdm.(ele).(src).^2)), ...
            abs(fpderr_sys.(ele).(src)(1)), ...
            mksz*fmsm.(src),fclr.(src),'filled', 'marker',fmrk.(src), ...
            'markeredgecolor',fclr.(src), ...
            'markerfacealpha',0.7, 'markeredgealpha',0)
    end %if
    
    

%     if any(SL)
%         flegs{end+1} = src;
%     end %if

end %for S
end %for E