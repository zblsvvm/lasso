Hello, 

Use this software to perform LASSO regression on input datasets. 



LOADING DATASET: 

Specify relative filepath to dataset by changing the location variable in data.py. 

Please ensure that dataset is vertical, n rows x m columns

Specify dataset size in data.py:
	set nof as the number of features/dimensions
	set y as the column number of output/y



RUNNING: 
Run main.py
Select minimisation algorithm then press enter
Select output type and press enter:

a Optimise --> Optimises function, outputs optimal lambda and corresponding polynomial coefficients and weights plus graphs of chi-squared vs lambda, y_predict vs y_observe


b  Residuals vs Lambda --> Produces  residuals vs lambda graph
 
c Evolution of coefficients --> Produces graph showing obtained coefficients vs lambda
 
d Dataset Size vs Scores --> Produces graph showing score vs dataset split size
 
e Coefficients vs Coefficients --> Produces graph showing relationship between coefficients 


Troubleshooting: 
If overflow or invalid value error is experienced for batch or gradient descent, change the u variable in line 46 of LinearRegression.py for batch gradient descent and line 126 for stochastic gradient descent. 

For faster simulations, consider a narrower range of lambdas or larger lambda values. 


Github repository: 
https://github.com/zblsvvm/lasso


