# Author: Seán McEntee
# purpose: To scrape the information about every car on donedeal using the text file generated previously. Also generates an Excel spreadsheet with the information on each car. 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import numpy as np
import pandas as pd

df = pd.read_csv("Donedeal_cars.txt", header=None) # Reading in text file containing links to each car on donedeal
links = np.array(df.iloc[:, 0])

count = 0
keyword = 'motorbikes'
keyword2 = 'new-car-for-sale'
keyword3 = 'commercials-for-sale' # removes ads, motorbikes, cars not yet for sale
indices = []
for i, link in enumerate(links):
    if keyword in link or keyword2 in link or keyword3 in link:
        indices.append(i)

indices = sorted(indices, reverse=True)
for i in indices:
    links = np.delete(links, i)

# options = Options()
# options.headless = True
# driver = webdriver.Chrome(options=options)
driver = webdriver.Chrome() # launch web browser

price_list = []; name_list = []; engine_list = []; body_type_list = []
tranmission_list = []; colour_list = []; mileage_list = []; year_list = []
owners_list = []; doors_list = []; seats_list = []; years_old_list = []
engine_type_list = []; engine_size_list = []; make_list = [] # lists for each parameter to be gained

count = 0
for i in range(len(links)):
    driver.get(links[i])
    time.sleep(0.1)
    html = driver.page_source

    soup = BeautifulSoup(html, 'lxml')
    
    info = soup.find_all('span', class_='meta-info__value')
    make = info[0].text.rstrip()[1:] # Removing newlines etc.
    year = info[2].text.rstrip()[1:]
    mileage = info[3].text.rstrip()[1:]
    engine_type = info[4].text.rstrip()[1:]
    tranmission = info[5].text.rstrip()[1:]
    body_type = info[6].text.rstrip()[1:]
    seats = info[7].text.rstrip()[1:]
    engine_size = info[8].text.rstrip()[1:].rsplit(' ', 1)[0] #Just taking first word
    owners = info[11].text.rstrip()[1:]
    colour = info[13].text.rstrip()[1:]
    doors = info[14].text.rstrip()[1:]

    no_price = soup.find('span', class_='price-value').text.rstrip()[2:] # Don;t want cars with unlisted prices

    if no_price != 'No price' and doors != '---' and colour != '---' and owners != '---' and engine_size != '---' and seats != '---' and body_type != '---' and tranmission != '---' and engine_type != '---' and make != '---': # removing entries with a missing parameter
        make_list.append(make)
        year_list.append(int(year))
        years_old_list.append(2021 - int(year))
        mileage_list.append(mileage)
        engine_type_list.append(engine_type)
        tranmission_list.append(tranmission)
        body_type_list.append(body_type)
        seats_list.append((seats))
        engine_size_list.append((engine_size))
        owners_list.append((owners))
        colour_list.append(colour)
        doors_list.append((doors))
        engine = engine_size + ' ' + engine_type
        engine_list.append(engine) # appending values to lists

        currency = soup.find('span', class_='currency ng-binding ng-scope').text[0]
        price = soup.find('span', class_='price ng-binding').text
        price_list.append(currency + price)

        name = soup.find('h1', class_='ng-binding').text
        name_list.append(name)

    count += 1
    if count % 10 == 0:
        print(count)
    # print('Price, Name, Engine, Body Type, Transmission, Colour, Mileage, Year, Owners, Doors, Seats, Years old, Engine Type, Engine Size, Make')
    # print(f'{currency + price}, {name}, {engine}, {body_type}, {tranmission}, {colour}, {mileage}, {int(year)}, {int(owners)}, {int(doors)}, {int(seats)}, {2021 - int(year)}, {engine_type}, {engine_size}, {make}')

d = {'Price': price_list, 'Name': name_list, 'Engine': engine_list, 'Bodytype': body_type_list, 'Transmission': tranmission_list, 'Colour': colour_list, 'Mileage': mileage_list, 'Year': year_list, 'Owners': owners_list, 'Doors': doors_list, 'Seats': seats_list, 'Years Old': years_old_list, 'Engine Type': engine_type_list, 'Engine Size': engine_size_list, 'Make': make_list} # Combining lists as a dictionary

df2 = pd.DataFrame(data=d) # COnverting dictionary to Pandas dataframe
df2.to_excel('DoneDeal_cars_9045_9901_thousand.xlsx') # Converting dataframe to Excel file (had to make multiple files due to errors arising when cars had been sold and taken down from the website
