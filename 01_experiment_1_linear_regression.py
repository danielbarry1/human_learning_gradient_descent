# this code performs the linear regression in experiment 1
# the supporting data can be found in the 
# "input_data_for_linear_regression_and_gradient_descent" 
# folder in "methods" on figshare

import pandas
import numpy as np
from sklearn import linear_model

def LinearRegressionComp():
	df=pandas.read_csv("data.csv")
	X = df[['x1','x2','x3']]
	y = df['y']
	regr = linear_model.LinearRegression(fit_intercept=True)
	regr.fit(X, y)
	print("regression coefficients + intercept: ",regr.coef_,regr.intercept_)
	return

LinearRegressionComp()