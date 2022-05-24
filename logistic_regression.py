import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import patsy
from sklearn.linear_model import LogisticRegression

test_data = pd.DataFrame(pd.read_csv("titanic.csv"))

#model_1
model_1 = smf.glm(formula='Survived ~ C(Pclass) * C(Sex)', data=test_data, family=sm.families.Binomial())
res = model_1.fit()
print(res.summary())

#model_2
y, X = patsy.dmatrices("Survived ~ C(Pclass) * C(Sex)", test_data)
model_2 = sm.Logit(y,X)
res = model_2.fit()
print(res.summary())

#model_3
y, X = patsy.dmatrices("Survived ~ C(Pclass) * C(Sex)", test_data)
new_X = pd.DataFrame(data=X, columns=X.design_info.column_names)
model_3 = LogisticRegression(penalty='none')
res = model_3.fit(new_X.loc[:, 'C(Pclass)[T.2]':], y)
res.coef_
res.intercept_
