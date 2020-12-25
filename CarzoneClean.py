import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir('C:/Users/lezuc/Desktop/')
df = pd.read_csv("cars.csv", comment='#')
print(df.head())
X1=df.iloc[ : , 0 ]
X2=df.iloc[ : , 2 ]
X=np.column_stack((X1, X2))
y=df.iloc[ : , 0 ]

df = pd.get_dummies(df)

x_colour = pd.get_dummies(df['Colour'])

x = df.iloc[ : , 1: ]

from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(x, y)




def ColourChange(colour, car):
    if colour in car:
        car = colour

for carColour in df.Colour:
    print(carColour)
    if "Blue" in carColour:
        carColour = "Blue"
    elif "Black" in carColour:
        carColour = "Black"
    elif "Yellow" in carColour:
        carColour = "Yellow"
    elif "Silver" in carColour:
        carColour = "Silver"
    elif "Grey" in carColour:
        carColour = "Grey"
    elif "Red" in carColour:
        carColour = "Red"
    elif "White" in carColour:
        carColour = "White"
    elif "Beige" in carColour:
        carColour = "Beige"

#df = dfx
df = df.reset_index(drop = True)
df = df[df.Seats.notna()]
df.to_csv(r'C:\Users\lezuc\Documents\carsForWork.csv')

df.loc[df['Colour'].str.contains('Black'), 'Colour'] = 'Black'
df.loc[df['Colour'].str.contains('Blue'), 'Colour'] = 'Blue'
df.loc[df['Colour'].str.contains('Red'), 'Colour'] = 'Red'
df.loc[df['Colour'].str.contains('Grey'), 'Colour'] = 'Grey'
df.loc[df['Colour'].str.contains('Yellow'), 'Colour'] = 'Yellow'
df.loc[df['Colour'].str.contains('Green'), 'Colour'] = 'Green'
df.loc[df['Colour'].str.contains('Metal'), 'Colour'] = 'Silver'
df.loc[df['Colour'].str.contains('Burgundy'), 'Colour'] = 'Red'
df.loc[df['Colour'].str.contains('Aqua'), 'Colour'] = 'Blue'
df.loc[df['Colour'].str.contains('White'), 'Colour'] = 'White'
df.loc[df['Colour'].str.contains('Silver'), 'Colour'] = 'Silver'
df.loc[df['Colour'].str.contains('Beige'), 'Colour'] = 'Beige'
df.loc[df['Colour'].str.contains('Bronze'), 'Colour'] = 'Bronze'
df.loc[df['Colour'].str.contains('Gold'), 'Colour'] = 'Gold'
df.loc[df['Colour'].str.contains('Brown'), 'Colour'] = 'Brown'
df.loc[df['Colour'].str.contains('Orange'), 'Colour'] = 'Orange'
df.loc[df['Colour'].str.contains('Navy'), 'Colour'] = 'Blue'


keepList = ['Black', 'Blue', 'Red', 'Grey', 'Yellow', 'Green', 'Silver', 'White', 'Beige', 'Bronze', 'Gold', 'Brown', 'Orange']
df = df[df.Colour.isin(keepList)]

Unlisted = ['POA']
df = df[~df.Price.isin(Unlisted)]

bins = [0, 0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9, 4.4, 4.9, 5.4, 5.9, 6.4, 6.9, 7.4, 7.9, 8.4]
labels = [0,0.5,1,1.5,2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
df['EngineSize'] = pd.cut(df['EngineSize'], bins=bins, labels=labels)

#df['EngineSize'] = dfx['EngineSize']

df['Make'] = df['Name']

df.Make = df.Make.str.split(" ").str.get(0)
