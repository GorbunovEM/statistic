#pip install stepwise-regression

import stepwise_regression
step_reg.backward_regression(X,Y,0.05)

lin_reg = LinearRegression()
rfe_mod = RFE(lin_reg, 5, step=1)
myvalues=rfe_mod.fit(X,Y)
