from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import requests
import xlsxwriter
import os
import pandas

path = 'C:\chromedriver.exe'

driver = webdriver.Chrome(path)
driver.get('http://asianfood.no/#login')
user = driver.find_element_by_id('username')
user.send_keys('omaralaissami84@gmail.com')
password = driver.find_element_by_id('password')
password.send_keys('sedora2005')
password.send_keys(Keys.RETURN)

urls = [
    'http://asianfood.no/ferske-varer',
    'http://asianfood.no/t%C3%B8rrvarer',
    'http://asianfood.no/restaurant-frukt--and--gr%C3%B8nt',
    'http://asianfood.no/forh%C3%A5ndbest--frukt--and--gr%C3%B8nt',
    'http://asianfood.no/frysevarer',
    'http://asianfood.no/non-food',
]

cnt = [
    2,
    49,
    5,
    20,
    13,
    6
]

cnti = 0
c = 0

for i, url in enumerate(urls):
    refs = []
    workbook = xlsxwriter.Workbook('C:\\Users\Omar\Desktop\\asia engros\\' + str(c) + '.xlsx')
    worksheet = workbook.add_worksheet()
    c += 1
    for j in range(1, cnt[i]):
        driver.get(url + '?pageID=' + str(j))
        products = driver.find_elements_by_class_name('Layout3Element')
        print(len(products))
        for p in products:
            try:
                h = p.find_element_by_class_name(
                    'AddHeaderContainer').find_element_by_tag_name('a').get_attribute(
                    'href')
                print(h)
                refs.append(h)
            except:
                continue

    for k in range(len(refs)):
        try:
            driver.get(refs[k])
            title = driver.find_element_by_class_name('heading-container').find_element_by_tag_name('h1').text
            price = driver.find_element_by_class_name('PriceLabel')
            price1 = price.get_attribute('data-priceincvat')
            price2 = price.get_attribute('data-priceexvat')
            try:
                img = driver.find_element_by_class_name('rsImg').get_attribute('src')
                r = requests.get(img)
                path = 'C:\\Users\Omar\Desktop\\asia engros\images\\' + str(cnti) + '.jpg'
                worksheet.(cnti, 3, str(cnti))
                with open(path, 'wb') as f:
                    f.write(r.content)
            except:
                worksheet.write(cnti, 3, 'not found')
            print(title)
            print(price1)
            print(price2)
            worksheet.write(cnti, 0, str(title))
            worksheet.write(cnti, 1, str(price1))
            worksheet.write(cnti, 2, str(price2))

            cnti += 1
        except:
            k = k - 1
    workbook.close()


