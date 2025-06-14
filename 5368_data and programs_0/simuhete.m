%--------------------------------------------------------------------------%
% SIMUHETE computes the part of the profits that is constant across        %
% simulations, and calls the routine SIMULOOP.m,  where the simulated      %
% profits are computed and the equilibria determined.                      %
%                                                                          %
%   function sand = simuhete(param)                                        %
%                                                                          %
% INPUTS                                                                   %
%   param:     This is the set of parameter values under consideration     %
%                                                                          %
% OUTPUTS                                                                  %
%   sand:      This is the distance function that is being minimized.      %
%                                                                          %
% Written by Federico Ciliberto and Elie Tamer, March 2003.                %
% When using this code or parts of it, please cite the following article:  %
% Federico Ciliberto and Elie Tamer, "Market Structure and Multiple        %
%          Equilibria in the Airline Industry," Econometrica, Vol. 77,     %
%          No. 6 (November, 2009), 1791-1828.                              %
%                                                                          %
%--------------------------------------------------------------------------%
                                                                           %
function sand = simuhete(param)                                            %
                                                                           %
global total r k  prob X rowX                                              %
global indexheter diffdummies  repindex                                    %
                                                                           %
%--------------------------------------------------------------------------%
% (1) DEFINE THE PARAMETERS                                                %
%--------------------------------------------------------------------------%
                                                                           %
paraconstant=param(1);                                                     % Constant
                                                                           %
paramX=param(2:9);                                                         % Exogenous Variables
                                                                           %
paraheterog1 = param(10);                                                  % Market Presence
                                                                           %
paraheterog2 = param(11);                                                  % Opportunity Cost
                                                                           %
parafirm1=param(12:17);                                                    % Competitive Effects
                                                                           %
%--------------------------------------------------------------------------%
% (2) COMPUTE THE EFFECT OF THE CONTROL VARIABLES                          %
%--------------------------------------------------------------------------%
                                                                           %
roleX=X(:,1:8)*paramX;                                                     %
                                                                           %
roleX=roleX*ones(1,k);                                                     %
                                                                           %
%--------------------------------------------------------------------------%
% (3) COMPUTE EFFECT OF HETEROGENEITY                                      %
%--------------------------------------------------------------------------%
                                                                           %
roleheter = X(:,9:14)*paraheterog1;                                        % Heterogeneity in Market Presence.
                                                                           %
roleheter=roleheter+X(:,15:20)*paraheterog2;                               % Heterogeneity in Opportunity Cost.
                                                                           %
%--------------------------------------------------------------------------%
% (4) COMPETITIVE EFFECTS                                                  %
%--------------------------------------------------------------------------%
                                                                           %
onothereffect=zeros(total*size(X,1),k);                                    % The sum of the competitive effects change by market structure.
                                                                           %     Dimension: (#markets*(2^#firms)) X #firms
                                                                           %                                          
for i=1:k                                                                  %
    effect=repindex(:,i)*parafirm1(i)*ones(1,k);                           %
    effect(:,i)=zeros(total*size(X,1),1);                                  % Makes sure that a firm does not negatively affect its own profit.
    onothereffect=onothereffect+effect;                                    % Sum over all the firms in the market.
end                                                                        %
clear effect                                                               %
                                                                           %
%--------------------------------------------------------------------------%
% (5) CONSTANT TERM                                                        %
%--------------------------------------------------------------------------%
                                                                           %
owneffect=[ones(total,k)*paraconstant];                                    %
                                                                           %
%--------------------------------------------------------------------------%
% (6) COMPUTES THE PART OF THE PROFIT THAT IS COMMON ACROSS SIMULATIONS    %
%--------------------------------------------------------------------------%
                                                                           %
common=roleX(indexheter)+roleheter(indexheter)+owneffect(diffdummies)+onothereffect;
                                                                           %
%--------------------------------------------------------------------------%
% (7) CALLS THE SIMULATION PROCEDURE WHERE THE LOOP ACROSS SIMULATION IS   %
% EXECUTED                                                                 %
%--------------------------------------------------------------------------%
                                                                           %
[meanupp,meanlow,temp] = simuloop(common);                                 %
                                                                           %
%--------------------------------------------------------------------------%
% (8) COMPUTES THE VALUE OF THE MINIMUM DISTANCE FUNCTION FOR THE          %
% PARAMETERS UNDER CONSIDERATION                                           %
%--------------------------------------------------------------------------%
                                                                           %
sand = ((prob(1:rowX,:)-meanupp).^2).*(prob(1:rowX,:)>meanupp)...          %
     +((prob(1:rowX,:) - meanlow).^2).*(prob(1:rowX,:)<meanlow);           % Check whether the empirical probability
                                                                           % is inside the lower and upper bounds.
                                                                           % If not, keep track of the difference.
                                                                           %    Dimension: # markets X 2^#firms
                                                                           %
%%%------------------------------------------------------------------------%
%%% (8a) THIS SECTION DEALS WITH THE POSSIBILITY THAT PURE STRATEGY        %
%%% EQUILIBRIA DO NOT EXIST IN ONE MARKET FOR ANY OF THE r SIMULATIONS.    %
%%%        Warning: this way to deal with the non-existence of pure        %
%%%        strategy equilibria works only if there are few                 %
%%%        market-simulation instances where pure strategy equilibria do   %
%%%        not exist, otherwise the routine might diverge, trying to choose
%%%        parameters that exclude as many market-simulations as possible. %
%%%        In our paper, there is enough variation in the exogenous        %
%%%        variables that we -never- observe a market-simulation instance  %
%%%        where there are no pure strategy equilibria.                    %
%%%------------------------------------------------------------------------%
                                                                           %
neverpure=min(isfinite(meanupp),[],2);                                     %
findneverpure=find(neverpure==0);                                          %
sizeneverpure=size(findneverpure,1);                                       %
if sizeneverpure~=0                                                        %
    sand(findneverpure,:)=[];                                              %
else                                                                       %
end                                                                        %
                                                                           %
%%%------------------------------------------------------------------------%
%%% (8b) THIS SECTION COMPUTES THE ACTUAL VALUE OF THE DISTANCE FUNCTION   %
%%%------------------------------------------------------------------------%
                                                                           %
sand=sum(sum(sand'));                                                      % Sum over all the markets and market structures
                                                                           % to get the distance function value.
%--------------------------------------------------------------------------%
% (9) SAVES THE RESULTS FROM THIS ITERATION, WHICH WILL BE USED LATER ON TO
% CONSTRUCT THE CONFIDENCE INTERVAL                                        %
%--------------------------------------------------------------------------%
                                                                           %
global sim_num                                                             %
sim_num=num2str(sim_num);                                                  %
load(['oldresultsasa',sim_num]);                                           %
                                                                           %
global iteration                                                           %
iteration=iteration+1;                                                     % Keeps track of the current number of iterations.
oldresultsasa=[oldresultsasa;[temp,sand,iteration,r,param']];              %
save(['oldresultsasa',sim_num], 'oldresultsasa')                           %
                                                                           %
end                                                                        %
                                                                           %