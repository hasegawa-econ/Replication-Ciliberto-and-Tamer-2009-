%--------------------------------------------------------------------------%
% MAKEINDEX is an auxiliary file. It creates the matrix of possible        %
% market structures for a given number of firms                            %
%                                                                          %
%   function [index]=makeindex(k)                                          %
%                                                                          %
% INPUTS                                                                   %
%   k:          Number of firms                                            %
%                                                                          %
% OUTPUTS                                                                  %
%   index:      Possible market structures                                 %
%               Dimension: 2^#firms X # firms                              %
%                                                                          %
% Written by Federico Ciliberto and Elie Tamer, March 2003.                %
% When using this code or parts of it, please cite the following article:  %
% Federico Ciliberto and Elie Tamer, "Market Structure and Multiple        %
%          Equilibria in the Airline Industry," Econometrica, Vol. 77,     %
%          No. 6 (November, 2009), 1791-1828.                              %
%                                                                          %
%--------------------------------------------------------------------------%
                                                                           %
function [index]=makeindex(k)                                              %
                                                                           %
total=2^k;                                                                 %
index=zeros(total,k);                                                      %
i=1;                                                                       %
for i=1:k                                                                  %
    ii=1;                                                                  %
    cc=1;                                                                  %
    c=total/(2^i);                                                         %
    while ii<=total                                                        %
        if cc <=c                                                          %
            index(ii,i)=1;                                                 %
            cc=cc+1;                                                       %
            ii=ii+1;                                                       %
        else                                                               %
            ii=ii+c;                                                       %
            cc=1;                                                          %
        end                                                                %
    end                                                                    %
end                                                                        %
                                                                           %
end                                                                        %
                                                                           %