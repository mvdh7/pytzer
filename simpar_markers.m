function [fmrk,fclr,fmsm] = simpar_markers(srcs)

mrks = repmat({'o' 'v' '^' '<' '>' 'sq' 'h' 'p' 'd'},1,3);
msms = repmat([ 1   1   1   1   1   1    1   3   1 ],1,3);
clrs = repmat([228,26,28; 55,126,184; 77,175,74; 152,78,163; 255,127,0; 
    166,86,40; 247,129,191; 153,153,153; 114 13 14] / 255,3,1);
for S = 1:numel(srcs)
    fmrk.(srcs{S}) = mrks{S};
    fclr.(srcs{S}) = clrs(S,:);
    fmsm.(srcs{S}) = msms(S);
end %for S

end %function simpar_markers
