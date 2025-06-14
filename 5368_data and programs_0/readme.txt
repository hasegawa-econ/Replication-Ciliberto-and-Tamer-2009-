
____________________________________________________________________________________________
                                                                             
This directory contains files that can be used to replicate the results in 
Column 2 of Table 3 in the article: Federico Ciliberto and Elie Tamer, "Market            
Structure and Multiple Equilibria in the Airline Industry," Econometrica,    
Vol. ZZZ, N. ZZZ., pp. ZZ-ZZ. The other columns can be easily replicated by
changing these files.

____________________________________________________________________________________________
                                                                             
The following files are included:                                            
start_values: This is the file of the starting values for the                
              minimization routine.                                          
                                                                             
mainhete.m:   This is the file that runs the minimization routine.           
              This file calls the file simuhete.m, which calls the file      
              simuloop.m                                                     
simuhete.m:   This file computes the common part across simulations of       
              the profits of the firms.                                      
simuloop.m:   This file finds the equilibria in each market, and             
              constructs the lower and upper bounds.                         
makeindex.m:  This is an auxiliary file, which constructs a matrix of        
              possible market structures for the given number of firms.      
                                                                             
confidencefcn.m: This is the file that computes the threshold that is        
                 needed to construct the confidence intervals.               
                                                                             
CilibertoTamerEconometrica.dta: This is the original dataset, whose          
                                construction is described in the Supplement  
                                to the paper.                                
____________________________________________________________________________________________

The steps to be followed to replicate the results are the following:

1) Take the original dataset, CilibertoTamerEconometrica.dta, and use it to
construct the first stage empirical probabilities. We discretize the 
variables and then run a simple non-parametric estimator. The discretization
can be done using quartiles of the continuous variables; using categorical
variables for some of the variables; or by dividing the support of the 
continuous variables by an appropriate number of intervals.

2) Then, one needs to set up the starting values. We use many starting values
one of which has all entries equal to zero. Other ones are constructed using
the results of simple probits, which are particularly useful to get good
starting values for the exogenous control variables.
The file start values contains the argmin for the specification in Column 2
of Table 3.

3) Then, one can run the code in matlab by running the function mainhete.m
This function calls the files simuhete.m, simuloop.m, makeindex.m.
Notice that we save the data of each iteration run in the course of the 
minimization. These parameter values will be used below to
construct the confidence intervals for the parameters.

4) After finding the minimum, one needs to construct the confidence interval.
To do that, one needs to conjecture a threshold point. We look at the distribution 
of the values that the distance function has taken over all the parameters used in 
the course of the minimization. We take the 25th percentile above the minimum found,
and use it as the initial threshold. We take all the parameters for which the objective
function is below this threshold.

5) Of all the parameters that we have saved and that are associated with
values of the distance function that are below the selected threshold, we
draw a random sample. One can use 200, 500, or 1000. 

6) Then, we construct R random subsamples of size 1/4 of the size of the original dataset.
One can set R=100, for example. This is the subsample size, and it is a good idea
to repeat all the steps here for many subsample sizes and make sure the
confidence regions do not change.

7) One now can run the file confidencefcn.m, which will return a number. This number 
must be divided by the sample size of the original dataset and summed to the
value of the distance function at the minimum. The result is the new threshold.
The paper contains description on what exactly goes into confidencefcn.m 

8) We usually iterate the above a few more times using the new threshold just found.
In theory, iterating a few times will help. But, we generally stopped after two or 
three iterations, or when the new threshold did not move away from the old one.

____________________________________________________________________________________________
