import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir('C:/Users/lezuc/Documents/')
df = pd.read_csv("carsForWork.csv")
df = pd.read_csv("DoneDeal_cars.csv")

############################

dfn = df
dfn = dfn.drop(columns=['Colour', 'Name', 'Engine', 'Year'])


from sklearn import preprocessing

v = dfn.select_dtypes(include=[np.number]) #returns a numpy array
scaler = preprocessing.StandardScaler()
v_scaled = scaler.fit_transform(v)
dfn[v.columns] = pd.DataFrame(v_scaled, columns=v.columns)

dfn = pd.get_dummies(dfn)

y=dfn.iloc[ : , 0 ]
x = dfn.iloc[ : , 1: ]

from sklearn.preprocessing import PolynomialFeatures
xpoly = PolynomialFeatures(3).fit_transform(x[['Years Old']])

dfn.insert(6, "Squared Years Old", xpoly[:,2])
dfn.insert(7, "Cubed Years Old", xpoly[:,3])

x = dfn.iloc[ : , 1: ]

############### DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor
mean_error=[]; std_error=[]; rsq = []
temp=[]; rtemp = []
model = DecisionTreeRegressor()
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
for train, test in kf.split(x):
    model.fit(x.iloc[train], y.iloc[train])
    ypred = model.predict(x.iloc[test])
    from sklearn.metrics import mean_squared_error
    temp.append(mean_squared_error(y[test],ypred))
    rtemp.append(model.score(x.iloc[test], y.iloc[test]))
mean_error.append(np.array(temp).mean())
std_error.append(np.array(temp).std())
rsq.append(np.array(rtemp).mean())

print(rtemp)
print(mean_error)
print(rsq)