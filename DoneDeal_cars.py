# Author: Seán McEntee
# purpose: to generate a text file containing links to the page of each car on the donedeal website
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

links = [] # to store link to each car on donedeal website
driver = webdriver.Chrome() # Launches webdriver
# Enter whatever URL you like
for i in range(2676): # Number of pages on website
    driver.get("https://www.donedeal.ie/cars?start=" + str(i * 28)) # 28 cars per page

    # let the code on their end run
    time.sleep(.01)
    # Save it as a variable
    html = driver.page_source
    # driver.quit()
    # And then just paste it right back into beautifulsoup!
    projects_soup = BeautifulSoup(html, 'lxml')

    each_car = projects_soup.find_all("li", class_="card-item")
    for car in each_car:
        title_elem = car.find('p', class_="card__body-title")
        url_item = car.find('a')['href']
        price_elem = car.find('p', class_='card__price')
        links.append(url_item) # append each url to links list
        if None in (title_elem, price_elem):
            continue

        # print(title_elem.text)
        # print(f'Link here: {url_item}')
        # print(price_elem.text)
        # print('\n')

with open('DoneDeal_cars.txt', 'w') as f:
    for link in links:
        f.write(f"{link}\n") # Writing links to text file 
