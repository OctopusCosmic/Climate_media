# This version is working on searching thousands of articles per keyword
# It could be run for several times to avoid robot suspicion
from bs4 import BeautifulSoup
from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
import pandas as pd
"/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/article_info"
def login_to_proquest(driver):
    # get access to webpage
    web_link = "https://www.proquest.com/usnews/advanced?accountid=13360"
    driver.get(web_link)

    # login to boiler account
    username = "" # input your username here
    password = ""  # input your password here
    driver.find_element_by_name("username").send_keys(username)
    driver.find_element_by_name("password").send_keys(password)
    driver.find_element_by_name("submit").click()

    after_login_url = driver.current_url
    return after_login_url

def search(driver,keyword1,keyword2):
    # check fulltext box
    driver.find_element_by_name("fullTextLimit").click()

    # wait for loading
    time.sleep(2)

    # check newspaper box
    driver.find_element_by_id("SourceType_Newspapers").click()

    # auto-type into keyword
    driver.find_element_by_name("queryTermField").send_keys(keyword1)
    driver.find_element_by_name("queryTermField_0").send_keys(keyword2)



    # click search button
    xpath_search = "//*[@id='searchToResultPage']"
    driver.find_element_by_xpath(xpath_search).click()
    # driver.find_element_by_name("searchToResultPage").click()
    time.sleep(5)
    search_results_url = driver.current_url

    return search_results_url



def main():
    # set up chrome driver
    driver = webdriver.Chrome("/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/chromedriver")
    # set load time to 10 seconds
    driver.set_page_load_timeout(60)
    # call login_to_proquest function to log in to proquest database
    after_login_url = login_to_proquest(driver)


    keyword1_list = ['"Cap and Trade"',
                     '"carbon tax"',
                     '"emissions tax"',
                     '"gas tax"',
                     '"System Benefit Fund"',
                     '"System Benefit Charge"',
                     '"RPS"',
                     '"Clean Energy Standard"',
                     '"Alternative Energy Standard"',
                     '"EERS"',
                     '"Energy efficiency standard"',
                     '"Demand-side management"',
                     '"Net Metering"']
    keyword1 = keyword1_list[12] # modify here!!!
    keyword2 = "environment"  # always related to environment
    #keyword2 = ""


    # create a dataframe to store information
    df = pd.DataFrame(columns=["Keyword", "Title", "Author", "Publication_title", "Place_of_publication",
                               "First_page", "Section", "Document_URL"])


    # call search function to set up advanced search
    search_results_url = search(driver, keyword1, keyword2)  # this url can be used for more articles in current keywords

    # search
    total = driver.find_element_by_id("pqResultsCount").text
    total_num = total.split()[0].replace(",", "")
    print("total_num: " + total_num)
    article_links = []
    while True:
        indexes = driver.find_elements_by_class_name("indexing")
        index_list = []
        for index in indexes:
            index_list.append(index.text)
        print(index_list)

        result_abstract_button_classname = "addFlashPageParameterformat_abstract"
        result_details_button_classname = "addFlashPageParameterformat_citation  "
        articles_in_page = driver.find_elements_by_class_name(result_abstract_button_classname)
        articles_in_page1 = driver.find_elements_by_class_name(result_details_button_classname)
        for article in articles_in_page:
            print(article.get_attribute("href"))
            article_links.append(article.get_attribute("href"))
        for article in articles_in_page1:
            print(article.get_attribute("href"))
            article_links.append(article.get_attribute("href"))

        if total_num not in index_list:
            # get more links from next page
            time.sleep(1)
            #next_page_button = driver.find_element_by_xpath('//*[@id="updateForm"]/nav/ul/li[7]/a/span[2]/span')
            #next_page_button = driver.find_element_by_xpath('//*[@id="updateForm"]/nav/ul/li[9]/a/span[2]/span')
            #next_page_button = driver.find_element_by_class_name("uxf-icon uxf-right-open-large")
            next_page_button = driver.find_element_by_xpath('//*[@id="updateForm"]/nav/ul/li[9]/a')
            #next_page_button = driver.find_element_by_xpath('//*[@id="updateForm"]/nav/ul/li[5]/a')
            #next_page_button = driver.find_element_by_xpath('//*[@id="updateForm"]/nav/ul/li[6]/a')
            next_page_button.click()
        else:
            break



    #print("total_num: ", total_num)

    # convert list to dataframe
    df_link = pd.DataFrame(article_links)
    # store dataframe as csv file
    df_link.to_csv("article_links_net_metering.csv") # modify here!!

