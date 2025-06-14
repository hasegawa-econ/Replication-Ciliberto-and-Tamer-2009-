%--------------------------------------------------------------------------%
% SIMULOOP sets up the loop to find the equilibria in each market structure
% for each simulation.                                                     %
%                                                                          %
%   function [meanupp,meanlow,temp] = simuloop(common)                     %
%                                                                          %
% INPUTS (market specific)                                                 %
%   common:     The common part across simulations - this includes the part
%               of the profit associated with the exogenous variables as   %
%               well as the 'competitive effects'.                         %
%               Dimension: (#markets*(2^#firms)) X #firms                  %
%                                                                          %
% OUTPUTS                                                                  %
%   meanupp:    Upper Bound Probabilities for each Market Structure in each
%               market. Dimension: # markets X 2^#firms                    %
%   meanlow:    Lower Bound Probabilities for each Market Structure in each
%               market. Dimension: # markets X 2^#firms                    %
%   temp:       Percentage of Market-Simulations where there is no pure    %
%               strategy equilibrium. Dimension: 1 X 1                     %
%                                                                          %
% NOTES                                                                    %
%               (i)   The loop is only over simulations, not over markets. %
%               (ii)  Markets are stacked together in matrix format.       %
%               (iii) In practice we never had to deal with the problem of %
%               non-existence of pure strategy equilibria. Here we propose %
%               one way to deal with that problem, explained below. Another
%               solution could be to introduce a penalty function.         %
%                                                                          %
% Written by Federico Ciliberto and Elie Tamer, March 2003.                %
% When using this code or parts of it, please cite the following article:  %
% Federico Ciliberto and Elie Tamer, "Market Structure and Multiple        %
%          Equilibria in the Airline Industry," Econometrica, Vol. 77,     %
%          No. 6 (November, 2009), 1791-1828.                              %
%                                                                          %
%--------------------------------------------------------------------------%
                                                                           %
function [meanupp,meanlow,temp] = simuloop(common)                         %
global epsi r k repindex rowX total indexheter                             %
                                                                           %
%--------------------------------------------------------------------------%
% (1) SETS UP MATRICES TO COMPUTE UPPER AND LOWER BOUNDS                   %
%--------------------------------------------------------------------------%
sumupp=zeros(rowX,total);                                                  % Matrix for 'numerators' of upper bound.
sumlow=sumupp;                                                             % Matrix for 'numerators' of lower bound.
                                                                           %
%--------------------------------------------------------------------------%
% (2) SETS UP ACCOUNTING VARIABLES FOR NON-EXISTENCE OF PURE STRAT. EQUIL. %
%--------------------------------------------------------------------------%
                                                                           %
totcount=r*ones(rowX,1);                                                   %
temp=zeros(rowX,1);                                                        %
                                                                           %
%--------------------------------------------------------------------------%
% (3) LOOPS OVER SIMULATIONS TO FIND EQUILIBRIA AND COMPUTE LOWER AND UPPER
% BOUNDS IN EACH MARKETS                                                   %
%--------------------------------------------------------------------------%
                                                                           %
for i=1:r                                                                  %
                                                                           %
%%%------------------------------------------------------------------------%
%%% (3A)  COMPUTES THE PROFITS IN EACH MARKET FOR ANY POSSIBLE MARKET      %
%%% STRUCTURE                                                              %
%%%------------------------------------------------------------------------%
                                                                           %
        epsitemp=epsi(:,1+k*(i-1):k*(i-1)+k);                              % Matrix of unobservables.
                                                                           %    Dimension: #markets X #firms
                                                                           %        
        epsitemp=epsitemp(indexheter);                                     % Expand matrix of unobservables to al market structures.
                                                                           %    Dimension: (#markets*(2^#firms))X #firms
                                                                           %        
        equil=epsitemp+common;                                             % These are the profits made by each firm in each market
                                                                           % configuration in each market. It sums the part that is 
                                                                           % common across simulations to the part that is specific
                                                                           % to one simulation.
                                                                           %                                                 
%%%------------------------------------------------------------------------%
%%% (3B)  LOOKS FOR THE EQUILIBRIA                                         %
%%%------------------------------------------------------------------------%
                                                                           %        
        equil=(equil>=0)==repindex;                                        %
        sumequil=sum(equil,2);                                             %
        vectorequil=(sumequil==k*ones(total*rowX,1));                      %
                                                                           %
%%%------------------------------------------------------------------------%
%%% (3C)  CHECKS FOR MULTIPLE EQUILIBRIA                                   %
%%%------------------------------------------------------------------------%
                                                                           %
        cumsumvectorequil= cumsum(vectorequil);                            %
        sumvectorequil=cumsumvectorequil(total:total:rowX*total);          %
        sumvectorequil(2:size(sumvectorequil,1),:)=diff(sumvectorequil);   % Counts number of equilibria in each market for this simulation.
                                                                           %    
%%%------------------------------------------------------------------------%
%%% (3D)  CONSTRUCTS THE LOWER AND UPPER BOUNDS                            %
%%%------------------------------------------------------------------------%
                                                                           %    
        upp=reshape(vectorequil',total,rowX)';                             % Organizes the information on equilibria in a matrix format.
                                                                           %     Dimension: # markets X 2^#firms
                                                                           %                                                              
        low=upp;                                                           %
                                                                           %
        lowtobezero=find(sumvectorequil>1);                                % Checks where there are multiple equilibria.
        low(lowtobezero,:)=0;                                              % Sets the simulation-specific lower bound equal to zero.
                                                                           %
%%%------------------------------------------------------------------------%
%%% (3E)  CHECKS FOR EXISTENCE OF THE PURE STRATEGY EQUILIBRIA             %
%%%------------------------------------------------------------------------%
                                                                           %
        upptobezero=find(sumvectorequil==0);                               %
        rowupptobezero=size(upptobezero,1);                                %
        if rowupptobezero~=0                                               %
            count=ones(rowX,1).*(sumvectorequil==0);                       %
            temp=temp+count;                                               %
            totcount=totcount-count;                                       %
        end                                                                %
                                                                           %        
%%%------------------------------------------------------------------------%
%%% (3F)  SUMS OVER SIMULATIONS TO COMPUTE THE NUMERATOR OF THE LOWER AND  %
%%% UPPER BOUND PROBABILITIES                                              %
%%%------------------------------------------------------------------------%
                                                                           %        
        sumupp=sumupp+upp;                                                 %
        sumlow=sumlow+low;                                                 %
        clear low upp sumequil rowupptobezero cumsumvectorequil            %
        clear count epsitemp sumvectorequil upptobezero lowtobezero        %
        clear vectorequil                                                  %
end                                                                        %
                                                                           %
%--------------------------------------------------------------------------%
% (4)  COMPUTES THE LOWER AND UPPER BOUNDS BY DIVIDING THE NUMERATOR       %
% COMPUTED ABOVE BY NUMBER OF SIMULATIONS                                  %
%--------------------------------------------------------------------------%
                                                                           %
meanlow=sumlow./(totcount*ones(1,total));                                  % Here we divide by the number of simulations-markets where
                                                                           %     there are pure strategy equilibria.
meanupp=sumupp./(totcount*ones(1,total));                                  %
temp=sum(temp)/(r*rowX);                                                   % Percentage of market-simulations where there are not pure
                                                                           %     pure strategy equilibria.
end                                                                        %
                                                                           %