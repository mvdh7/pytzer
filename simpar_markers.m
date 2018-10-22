function [fmrk,fclr,fmsm] = simpar_markers(srcs)

mrks = repmat({'o' 'v' '^' '<' '>' 'sq' 'h' 'p' 'd'},1,4);
msms = repmat([ 1   1   1   1   1   1    1   2   1 ],1,4);
clrs = repmat([228,26,28; 46,139,87; 147,112,219; 255,127,0;
    160,82,45; 255,105,180; 153,153,153; 0 0 205; 0 191 255;
    75,0,130; 154,205,50; 47,79,79] / 255,4,1);
for S = 1:numel(srcs)
    fmrk.(srcs{S}) = mrks{S};
    fclr.(srcs{S}) = clrs(S,:);
    fmsm.(srcs{S}) = msms(S);
end %for S

end %function simpar_markers
