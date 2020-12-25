# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 17:57:09 2020

@author: lezuc
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir('C:/Users/lezuc/Documents/')
df = pd.read_csv("carsForWork.csv")
df = pd.read_csv("DoneDeal_cars.csv")

dfx = df
dfx = dfx.drop(columns=['Name', 'Engine'])

dfx = pd.get_dummies(dfx)

y=dfx.iloc[ : , 0 ]
x = dfx.iloc[ : , 1: ]



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

model = LinearRegression().fit(Xtrain, ytrain)
model.score(Xtest, ytest)

from sklearn.metrics import mean_squared_error
ypred = model.predict(Xtest)
mean_squared_error(ytest, ypred)

############ MODEL 2

df_noColour = df
df_noColour = df_noColour.drop(columns=['Colour', 'Name', 'Engine', 'Year'])

df_noColour = pd.get_dummies(df_noColour)

y=df_noColour.iloc[ : , 0 ]
x = df_noColour.iloc[ : , 1: ]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

model = LinearRegression().fit(Xtrain, ytrain)
model.score(Xtest, ytest)

from sklearn.metrics import mean_squared_error
ypred = model.predict(Xtest)
mean_squared_error(ytest, ypred)

############ MODEL 3

df_poly = df
df_poly = df_poly.drop(columns=['Colour', 'Name', 'Engine', 'Year'])

df_poly = pd.get_dummies(df_poly)

y= df_poly.iloc[ : , 0 ]
x = df_poly.iloc[ : , 1: ]

from sklearn.preprocessing import PolynomialFeatures
xpoly = PolynomialFeatures(3).fit_transform(x[['Years Old']])

df_poly.insert(6, "Squared Years Old", xpoly[:,2])
df_poly.insert(7, "Cubed Years Old", xpoly[:,3])

#xpoly = PolynomialFeatures(2).fit_transform(x[['Mileage']])

#df_poly.insert(2, "Squared Mileage", xpoly[:,2])

x = df_poly.iloc[ : , 1: ]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

model = LinearRegression().fit(Xtrain, ytrain)
model.score(Xtest, ytest)

from sklearn.metrics import mean_squared_error
ypred = model.predict(Xtest)
mean_squared_error(ytest, ypred)

############ MODEL 4

dfn = df
dfn = dfn.drop(columns=['Colour', 'Name', 'Engine', 'Year'])
dfn = pd.get_dummies(dfn)

dfn = (dfn - dfn.min()) / (dfn.max() - dfn.min())

y=dfn.iloc[ : , 0 ]
x = dfn.iloc[ : , 1: ]

from sklearn.preprocessing import PolynomialFeatures
xpoly = PolynomialFeatures(3).fit_transform(x[['Years Old']])

dfn.insert(6, "Squared Years Old", xpoly[:,2])
dfn.insert(7, "Cubed Years Old", xpoly[:,3])

x = dfn.iloc[ : , 1: ]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

model = LinearRegression().fit(Xtrain, ytrain)
model.score(Xtest, ytest)

from sklearn.metrics import mean_squared_error
ypred = model.predict(Xtest)
print(mean_squared_error(ytest, ypred))
print(ytest, ypred)

############ MODEL 5

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

#xpoly = PolynomialFeatures(2).fit_transform(x[['Doors']])

#dfn.insert(2, "Squared Mileage", xpoly[:,2])

x = dfn.iloc[ : , 1: ]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

model = LinearRegression().fit(Xtrain, ytrain)
print(ytrain)
print(model.score(Xtest, ytest))

from sklearn.metrics import mean_squared_error
ypred = model.predict(Xtest)
print(mean_squared_error(ytest, ypred))

############# LINEAR

mean_error=[]; std_error=[]; rsq = []
temp=[]; rtemp = []
model = LinearRegression()
from sklearn.model_selection import KFold
kf = KFold(n_splits=4)
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

############# RIDGE

mean_error=[]; std_error=[]; rsq = []

Ci_range = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
for Ci in Ci_range:
    temp=[]; rtemp = []
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1/(2*Ci))
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=25)
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

############# LASSO

mean_error=[]; std_error=[]; rsq = []

Ci_range = [0.00001, 0.0001, 0.001, 0.01, 0.1]
for Ci in Ci_range:
    temp=[]; rtemp = []
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=Ci)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=25)
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

############### BAGGING

from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
mean_error=[]; std_error=[]; rsq = []
temp=[]; rtemp = []
model = BaggingRegressor()#))
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

############### RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor
mean_error=[]; std_error=[]; rsq = []
temp=[]; rtemp = []
model = RandomForestRegressor()
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