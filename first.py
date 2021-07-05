from warnings import catch_warnings

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import requests
import xlsxwriter
import os

path = 'C:\chromedriver.exe'

workbook = xlsxwriter.Workbook('C:\\Users\Omar\Desktop\ww\elite engros.xlsx')
worksheet = workbook.add_worksheet()

driver = webdriver.Chrome(path)
driver.get('https://eliteengros.mamutweb.com/Shop/List/(ny)/226/1')
email = driver.find_element_by_id('Username')
email.send_keys('omaralaissami84@gmail.com')
password = driver.find_element_by_id('Password')
password.send_keys('sedora2005')
password.send_keys(Keys.RETURN)

catagories = driver.find_elements_by_class_name('breadcrumbLink')
cnt = 0
c = 0

for i in range(len(catagories)):
    if i < 4:
        continue
    t = catagories[i].get_attribute('href')
    driver.get(t)
    # time.sleep(3)
    workbook = xlsxwriter.Workbook('C:\\Users\Omar\Desktop\ww\\' + str(c) + '.xlsx')
    worksheet = workbook.add_worksheet()
    c += 1
    while True:
        products = driver.find_elements_by_class_name('productListAltRow')
        for i, product in enumerate(products):
            if i % 2 != 0:
                continue
            try:
                name = product.find_element_by_class_name('productListProductName').text
                price = product.find_element_by_class_name('productPrice')
                grossPrice = price.get_attribute('data-grossprice')
                netPrice = price.get_attribute('data-netprice')
                try:
                    img = product.find_element_by_class_name('productImage').get_attribute('src')
                    r = requests.get(img)
                    path = 'C:\\Users\Omar\Desktop\ww\ww\\' + str(cnt) + '.jpg'
                    worksheet.write(cnt, 3, str(cnt))
                except:
                    worksheet.write(cnt, 3, 'not found')

                worksheet.write(cnt, 0, name)
                worksheet.write(cnt, 1, grossPrice)
                worksheet.write(cnt, 2, netPrice)

                cnt += 1
                if not os.path.exists(path):
                    with open(path, 'wb') as f:
                        f.write(r.content)
                    print(name)
            except:
                continue
        print(len(products))
        products = driver.find_elements_by_class_name('productListRow')

        for i, product in enumerate(products):
            if i % 2 != 0:
                continue
            try:
                name = product.find_element_by_class_name('productListProductName').text
                price = product.find_element_by_class_name('productPrice')
                grossPrice = price.get_attribute('data-grossprice')
                netPrice = price.get_attribute('data-netprice')

                try:
                    img = product.find_element_by_class_name('productImage').get_attribute('src')
                    r = requests.get(img)
                    path = 'C:\\Users\Omar\Desktop\ww\ww\\' + str(cnt) + '.jpg'
                    worksheet.write(cnt, 3, str(cnt))
                except:
                    worksheet.write(cnt, 3, 'not found')

                worksheet.write(cnt, 0, name)
                worksheet.write(cnt, 1, grossPrice)
                worksheet.write(cnt, 2, netPrice)

                cnt += 1
                if not os.path.exists(path):
                    with open(path, 'wb') as f:
                        f.write(r.content)
                    print(name)
            except:
                continue
        print(len(products))
        try:
            h = driver.find_element_by_class_name('productListTableHeader').find_element_by_class_name(
                'pagingButtonNext').get_attribute('href')
            if h is None:
                break
            driver.get(h)
        except:
            break
    catagories = driver.find_elements_by_class_name('breadcrumbLink')
    workbook.close()

driver.close()
workbook.close()
