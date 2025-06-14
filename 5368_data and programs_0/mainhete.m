%--------------------------------------------------------------------------%
% MAINHETE loads the dataset and sets up the minimization routine. The     %
% objective function is defined in the file SIMUHETE.m. The file           %
% SIMULOOP.m runs the simulation routine from within the file SIMUHETE.m.  %
% The file SIMULOOP.m runs the simulation routine.                         %
%                                                                          %
%   function mainhete()                                                    %
%                                                                          %
% NOTES                                                                    %
%       (i) This is written as a function because we later on compile the  %
%       files in an unique executable. However, this can be run            %
%       interactively within matlab. In that case the first line, function %
%       mainhete(), can be commented out.                                  %
%       (ii)  The first stage probabilities can estimated beforehand. In   %
%       our paper, we have estimated them by using a multinomial logit     %
%       model in Stata or by discretizing the variables and then running a %
%       simple non-parametric frequency estimator.                         %    
%       (iii) The simulated errors are drawn beforehand. This is because it%
%       is much easier to deal with airport specific random unobservables. %
%                                                                          %
% Written by Federico Ciliberto and Elie Tamer, March 2003.                %
% When using this code or parts of it, please cite the following article:  %
% Federico Ciliberto and Elie Tamer, "Market Structure and Multiple        %
%          Equilibria in the Airline Industry," Econometrica, Vol. 77,     %
%          No. 6 (November, 2009), 1791-1828.                              %
%                                                                          %
%--------------------------------------------------------------------------%
                                                                           %
function mainhete()                                                        %
                                                                           %
%--------------------------------------------------------------------------%
% (1) READS IN THE INITIAL VALUES OF THE PARAMETERS AND AN INDICATOR THAT  %
% CAN BE ASSOCIATED TO THEM                                                %
%--------------------------------------------------------------------------%
                                                                           %
start_values=textread('start_values');                                     % This is a matrix, whose first row is an indicator that is associated
                                                                           % with a particular starting value for the parameters and the second
                                                                           % row is the parameter vector for the starting values.
                                                                           % Dimension: 2 X (# parameters)
                                                                           %
global sim_num                                                             %
sim_num=num2str(start_values(1,1));                                        % Records particular starting value.
                                                                           %
param0=start_values(2,:)';                                                 % Starting values for the parameters.
                                                                           %
%--------------------------------------------------------------------------%
% (2) LOADS DATA                                                           %
%--------------------------------------------------------------------------%
                                                                           %
load marketdata.raw;                                                       %
                                                                           %
datairline=marketdata;                                                     % Allows for simple changes in the dataset being used.
                                                                           %
clear marketdata;                                                          %
                                                                           %
%--------------------------------------------------------------------------%
% (3) DEFINE VARIABLE COUNTING THE NUMBER OF FIRMS, THE MATRIX OF POSSIBLE %
% MARKET STRUCTURES                                                        %
%--------------------------------------------------------------------------%
                                                                           %
global index total k                                                       %
k=6;                                                                       % Number of firms.
                                                                           %
index=makeindex(k);                                                        % Matrix of possible market structures.
                                                                           % Dimension: (2^#firms) X #firms
                                                                           %
total=size(index,1);                                                       % This is the number of possible market structures = 2^#firms.
                                                                           %
%--------------------------------------------------------------------------%
% (4) SET THE NUMBER OF SIMULATIONS                                        %
%--------------------------------------------------------------------------%
                                                                           %
global r                                                                   %
r=100;                                                                     %
                                                                           %
%--------------------------------------------------------------------------%
% (5) DEFINE VARIABLES ASSOCIATED WITH DATA                                %
%--------------------------------------------------------------------------%
                                                                           %
global X rowX                                                              %
coldatairline=size(datairline,2);                                          %
                                                                           %
X=datairline(:,k+1:coldatairline-2*r);                                     % Exogenous regressors.
                                                                           % Dimension: # markets X # regressors
                                                                           %
colX=size(X,2);                                                            % Number of exogenous regressors.
                                                                           %
y=datairline(:,1:k);                                                       % Observed market structures. This vector is not used in this function
                                                                           % as we wrote it. It would be used if we were to run the first stage 
                                                                           % regression within this matlab function.
                                                                           % Dimension: # markets X # firms
                                                                           %
rowX=size(X,1);                                                            % # markets
                                                                           %
%--------------------------------------------------------------------------%
% (6) FIRST STAGE EMPIRICAL PROBABILITIES                                  %
%--------------------------------------------------------------------------%
                                                                           %
global prob                                                                %
load conddensityorder.raw                                                  % Loads empirical probabilities.
prob=conddensityorder;                                                     %
                                                                           %
%--------------------------------------------------------------------------%
% (7) DEFINE A COUNTING VARIABLE TO KEEP TRACK OF HOW MANY ITERATIONS HAVE %
% BEEN EXECUTED                                                            %
%--------------------------------------------------------------------------%
                                                                           %
global iteration                                                           %
iteration=0;                                                               %
                                                                           %
%--------------------------------------------------------------------------%
% (8) LOAD THE SIMULATED UNOBSERVABLES                                     %
%--------------------------------------------------------------------------%
                                                                           %
global epsi                                                                %
load firmmarketerrs.raw                                                    %
                                                                           %
epsi=firmmarketerrs(:,1:r*k);                                              % Firm specific unobservables. We draw outside of this program (in Stata).
                                                                           % # markets X (# simulations * #firms)
                                                                           %
epsimarket=firmmarketerrs(:,r*k+1:r*k+r);                                  % Market specific unobservables.
                                                                           % Dimension: # markets X # simulations
                                                                           %
epsimarket=kron(epsimarket,ones(1,k));                                     % Assigns the market unobservable to each of the k firms.
                                                                           % Dimension: # markets X (# simulations * #firms)
                                                                           %
epsi=epsi+epsimarket;                                                      % Adds market and firm unobservables.
                                                                           %
AirRndErr=datairline(:,k+colX+1:k+colX+2*r);                               % Gets the airport specific errors.
                                                                           % Dimension: # markets X # simulations * 2 (2 because of orig and dest)
                                                                           %
for i=1:r                                                                  %
    epsiairport(:,i)=AirRndErr(:,i)+AirRndErr(:,r+i);                      % Sums the origin and destination errors.
                                                                           % Dimension: # markets X # simulations
                                                                           %
end                                                                        %
clear i                                                                    %
                                                                           %
epsiairport=kron(epsiairport,ones(1,k));                                   % Assigns the airport specific errors to each firm.
                                                                           % Dimension: # markets X (# simulations * #firms)
                                                                           %
epsi=epsi+epsiairport;                                                     % Final set of unobservables.
                                                                           % Dimension: # markets X (# simulations * #firms)
                                                                           %
%--------------------------------------------------------------------------%
% (9) CONSTRUCTS INDEXES THAT PERMIT TO AVOID HAVING TO LOOP OVER MARKETS  %
%--------------------------------------------------------------------------%
                                                                           %
global indexheter                                                          %
tempheter=reshape(1:k*size(X,1),size(X,1),k);                              %
indexheter=zeros(size(X,1)*total,k);                                       %
for j=1:size(X,1)                                                          %
    indexheter((j-1)*total+1:(j-1)*total+total,:)=repmat(tempheter(j,:),total,1);                                                       %
end                                                                        %
clear tempheter                                                            %
                                                                           %
global repindex                                                            %
repindex=repmat(index,size(X,1),1);                                        %
                                                                           %
global diffdummies                                                         %
diffdummies=repmat(reshape(1:total*k,total,k),size(X,1),1);                %
                                                                           %
%--------------------------------------------------------------------------%
% (10) CREATE A MATLAB FILE THAT STORES THE RESULTS FROM EACH ITERATION    %
%--------------------------------------------------------------------------%
                                                                           %
oldresultsasa=[0,0,iteration,r,param0'];                                   %
save(['oldresultsasa',sim_num],'oldresultsasa')                            %
                                                                           %
%--------------------------------------------------------------------------%
% (11) SIMULATED ANNEALING                                                 %
%--------------------------------------------------------------------------%
                                                                           %
lb = ones(1,17)*(-30);                                                     % Minimum values that the parameters can take.
ub = ones(1,17)*30;                                                        % Maximum values that the parameters can take.
                                                                           %
[param,fval] = simulannealbnd(@simuhete,param0,lb, ub);                    % Here, it is useful to try different temperatures, different schedules
                                                                           % for the changes in temperatures, and so on, especially when starting 
                                                                           % the miminization process.
                                                                           %
end                                                                        %
                                                                           %