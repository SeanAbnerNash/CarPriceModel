# Author: Seán McEntee
# Purpose: Combine all individual excel spreadsheets to make one final spreadsheet. Also want to convert all currencies to euro and mileage to miles
import pandas as pd
import numpy as np

#Combining all spreadsheets that were made previously
df1 = pd.read_excel('DoneDeal_cars_1000.xlsx')

df2 = pd.read_excel('DoneDeal_cars_1-3_thousand.xlsx')

df3 = pd.read_excel('DoneDeal_cars_3-5_thousand.xlsx')

df4 = pd.read_excel('DoneDeal_cars_5-8856_thousand.xlsx')

df5 = pd.read_excel('DoneDeal_cars_8856_9041_thousand.xlsx')

df6 = pd.read_excel('DoneDeal_cars_9045_9901_thousand.xlsx')

df_final = df1.append(df2, ignore_index=True).append(df3, ignore_index=True).append(df4, ignore_index=True).append(df5, ignore_index=True).append(df6, ignore_index=True)

df_final = df_final.drop_duplicates() # removing duplicates from dataset


mileage_indices = []
for i in range(len(df_final)):
    if df_final['Mileage'][i] == '---':
        mileage_indices.append(i) # finding mileage entries which are empty and removing the corresponding rows

df_final = df_final.drop(mileage_indices)
df_final.index = range(len(df_final))
df_final.iloc[:,0] = np.arange(len(df_final))

mileage_miles_indices = []
price_pound_indices = []
for i in range(len(df_final)): 
    if 'mi' in df_final['Mileage'][i]:
        mileage_miles_indices.append(i)
    if '£' in df_final['Price'][i]:
        price_pound_indices.append(i)
    df_final['Mileage'][i] = df_final['Mileage'][i].rsplit(' ', 1)[0]
    df_final['Mileage'][i] = int(df_final['Mileage'][i].replace(',', ''))
    
    df_final['Price'][i] = df_final['Price'][i][1:]
    df_final['Price'][i] = int(df_final['Price'][i].replace(',', ''))

for i in mileage_miles_indices:
    df_final['Mileage'][i] = round(1.60934 * df_final['Mileage'][i]) # Converting miles to km 

for i in price_pound_indices:
    df_final['Price'][i] = round(1.10 * df_final['Price'][i]) # COnverting pounds to euro

df_final.to_excel('DoneDeal_cars_final.xlsx') # writing to final Excel file
