import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

pd.set_option('display.max_columns', None)

def login_to_proquest(driver,duo, web_link):
    # get access to webpage
    driver.get(web_link)

    # login to boiler account
    username = "zhan3447" # input your username here
    password = "0107," + duo  # input your password here
    driver.find_element_by_name("username").send_keys(username)
    driver.find_element_by_name("password").send_keys(password)
    driver.find_element_by_name("submit").click()

    after_login_url = driver.current_url
    return after_login_url

def add_info(row, df, titles_text, contents_text, count, keyword1):
    #df = pd.DataFrame(columns=["Keyword", "Title", "Author", "Publication_title", "Publication_date",
    #                           "Place_of_publication", "First_page", "Section", "Document_URL"])

    newObs = ["NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]##
    idx = list(df.columns)##
    # add this article information to a row of dataframe
    newObs[0] = keyword1
    for i in range(count):
        if (titles_text[i] == "Title"):
            newObs[1] = contents_text[i]
        if (titles_text[i] == "Author"):
            newObs[2] = contents_text[i]
        if (titles_text[i] == "Publication title"):
            newObs[3] = contents_text[i]
        if (titles_text[i] == "Publication date"):
            newObs[4] = contents_text[i]
        if (titles_text[i] == "Place of publication"):
            newObs[5] = contents_text[i]
        if (titles_text[i] == "First page"):
            newObs[6] = contents_text[i]
        if (titles_text[i] == "Section"):
            newObs[7] = contents_text[i]
        if (titles_text[i] == "Document URL"):
            newObs[8] = contents_text[i]


    pd.set_option('display.max_columns', None)

    newExample = pd.Series(newObs, name=row, index=idx)##
    df = df.append(newExample)##

    return df
def merge_files():
    keywords = ["alternative_energy_standard",
                "cap_and_trade",
                "carbon_tax",
                "clean_energy_standard",
                "demand-side_management",
                "EERS",
                "emissions_tax",
                "energy_efficiency_standard",
                "gas_tax",
                "net_metering",
                "RPS",
                "system_benefit_charge",
                "system_benefit_fund"]
    filenames = ["article_links_alternative_energy_standard.csv",
                 "article_links_cap_and_trade.csv",
                 "article_links_carbon_tax.csv",
                 "article_links_clean_energy_standard.csv",
                 "article_links_demand-side_management.csv",
                 "article_links_EERS.csv",
                 "article_links_emissions_tax.csv",
                 "article_links_energy_efficiency_standard.csv",
                 "article_links_gas_tax.csv",
                 "article_links_net_metering.csv",
                 "article_links_RPS.csv",
                 "article_links_system_benefit_charge.csv",
                 "article_links_system_benefit_fund.csv"]

    path = "/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/"
    df = pd.DataFrame()
    for i in range(len(filenames)):
        filename = filenames[i]
        data = pd.read_csv(path + filename)
        data = data.rename(columns={"Unnamed: 0": "index", "0": "link"})
        data = data.drop(columns=["index"])
        data["keyword"] = keywords[i]
        df = df.append(data)

    df = df.reset_index() # this dataframe store information about links for all keywords
    df.to_csv("article_links.csv")
def get_links_from_csv(path, filename):
    df = pd.read_csv(path + filename)
    df = df.drop("Unnamed: 0", axis=1)

    # convert columns to list
    article_links = df["article_link"].tolist()
    #article_links = df["link"].tolist()
    #keyword_list = df["keyword"].tolist() # if having keyword column

    print("number of articles: " + str(len(article_links)))
    return article_links#, keyword_list

def main_with_keyword():
    # read all article links from csv file
    path = "/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/"
    filename = "article_links.csv"
    article_links, keyword_list = get_links_from_csv(path,filename)

    # initialize value
    row = 0
    gap = 40 # Since a duo code only valid for 40 articles, further the account has been logged out
    low = row
    high = row + gap

    # while-loop for login for multiple times
    while (row < len(article_links)):
        # prompt user for duo password
        duo = input("password is: ")
        # set up chrome driver
        driver = webdriver.Chrome("/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/chromedriver_94")
        # set load time to 30 seconds
        driver.set_page_load_timeout(30)
        # call login_to_proquest function to log in to proquest database
        web_link = "https://www.proquest.com/usnews/advanced?accountid=13360"
        after_login_url = login_to_proquest(driver,duo,web_link)

        # initialize a dataframe for upcoming 40 articles' info
        df1 = pd.DataFrame(columns=["Keyword", "Title", "Author", "Publication_title", "Place_of_publication",
                                    "Publication_date","First_page", "Section", "Document_URL"])

        # while-loop for scraping from 40 articles' link
        while (low <= row < high and row < len(article_links)):
            # get current link from list
            link = article_links[row]
            # get current keyword from list
            keyword1 = keyword_list[row]
            # open current link
            driver.get(link)

            # start scraping information from Details table
            # get pagesource start to do bs4
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            titles_text = []
            contents_text = []
            # find tags that represent each row
            tag1s = soup.find_all("div", class_="display_record_indexing_row")
            # loop through all rows
            for each in tag1s:
                # get first tag with fieldname infomation
                temp_tag = each.find_next("div", class_="display_record_indexing_fieldname")
                # grab text as title
                title = temp_tag.get_text()
                text = each.get_text()
                # get content
                content = text.replace(title, "")
                titles_text.append(title)
                contents_text.append(content)

            print(titles_text)
            print(contents_text)

            # get the number of rows in the Details table
            count = len(titles_text)

            # call add_info function to add information in an article
            df1 = add_info(row, df1, titles_text, contents_text, count, keyword1)
            # update row
            row += 1

        print("row: " + str(row) + " (" + str(low) + ", " + str(high) + ")")

        # append info of 40 articles to the file
        df1.to_csv("new_climate_media_14000.csv", mode='a', header=False)

        # update lower and upper bound
        low += gap
        high += gap
        # quit current driver
        driver.quit()
def add_info_without_keyword(row, df, titles_text, contents_text, count):
    #df = pd.DataFrame(columns=["Title", "Author", "Publication_title", "Publication_date",
    #                           "Place_of_publication", "First_page", "Section", "Document_URL"])

    newObs = ["NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]##
    idx = list(df.columns)##
    # add this article information to a row of dataframe
    #newObs[0] = keyword1
    for i in range(count):
        if (titles_text[i] == "Title "):
            newObs[0] = contents_text[i]
        if (titles_text[i] == "Author "):
            newObs[1] = contents_text[i]
        if (titles_text[i] == "Publication title "):
            newObs[2] = contents_text[i]
        if (titles_text[i] == "Publication date "):
            newObs[3] = contents_text[i]
        if (titles_text[i] == "Place of publication "):
            newObs[4] = contents_text[i]
        if (titles_text[i] == "First page "):
            newObs[5] = contents_text[i]
        if (titles_text[i] == "Section "):
            newObs[6] = contents_text[i]
        if (titles_text[i] == "Document URL "):
            newObs[7] = contents_text[i]


    pd.set_option('display.max_columns', None)

    newExample = pd.Series(newObs, name=row, index=idx)##
    df = df.append(newExample)##

    return df
def make_sure_in_abstract_url(driver):
    html1 = driver.page_source
    soup1 = BeautifulSoup(html1, "html.parser")
    # we assume the full text tab is the first tag we find
    # abstract tab
    tag_abstract = soup1.find("li", id="tab-AbstractRecord-null")
    mark = "abstract detail"
    if tag_abstract == None:
        tag_abstract = soup1.find("li", id = "tab-Record-null")
        mark = "detail"
    # if given attribute is false
    if tag_abstract["aria-selected"] == "false":
        if mark == "abstract detail":
            driver.find_element_by_id("tab-AbstractRecord-null").click()
        elif mark == "detail":
            driver.find_element_by_id("tab-Record-null").click()

def driver_setup():
    # set up chrome driver
    driver = webdriver.Chrome("/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/chromedriver_v94")
    driver.set_page_load_timeout(60)
    web_link = "https://www.proquest.com/usnews/advanced?accountid=13360"
    duo = input("Password is: ")
    after_login_url = login_to_proquest(driver, duo, web_link)
    return driver

def main():
    # get all article links from full text links
    path = "/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/hand_coded_articles/"
    filename = "hand_coded_article_links.csv"
    article_links = get_links_from_csv(path, filename)

    # initialize value
    row = 0
    gap = 40  # Since a duo code only valid for 40 articles, further the account has been logged out
    low = row
    high = row + gap

    # while-loop for login for multiple times
    while (row < len(article_links)):
        # set up driver and login in to proquest database
        driver = driver_setup()

        # initialize a dataframe for upcoming 40 articles' info
        df1 = pd.DataFrame(columns=["Title", "Author", "Publication_title", "Publication_date",
                                    "Place_of_publication", "First_page", "Section", "Document_URL"])

        # while-loop for scraping from 40 articles' link
        while (low <= row < high and row < len(article_links)):
            # get current link from list
            abstract_link = article_links[row]
            # direct to abstract link
            driver.get(abstract_link)
            # from fulltext link to abstract detail link
            make_sure_in_abstract_url(driver)

            # start scraping information from Details table
            # get pagesource start to do bs4
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            titles_text = []
            contents_text = []
            # find tags that represent each row
            tag1s = soup.find_all("div", class_="display_record_indexing_row")
            # loop through all rows
            for each in tag1s:
                # get first tag with fieldname infomation
                temp_tag = each.find_next("div", class_="display_record_indexing_fieldname")
                # grab text as title
                title = temp_tag.get_text()
                text = each.get_text()
                # get content
                content = text.replace(title, "")
                titles_text.append(title)
                contents_text.append(content)

            print(titles_text)
            print(contents_text)

            # get the number of rows in the Details table
            count = len(titles_text)

            # call add_info function to add information in an article
            df1 = add_info_without_keyword(row, df1, titles_text, contents_text, count)
            # update row
            row += 1

        print("row: " + str(row) + " (" + str(low) + ", " + str(high) + ")")

        # append info of 40 articles to the file
        df1.to_csv(path + "hand_coded_info.csv", mode='a', header=False)

        # update lower and upper bound
        low += gap
        high += gap
        # quit current driver
        driver.quit()



main()

def test():
    full_text_link = "https://search.proquest.com/docview/1748840816?accountid=13360"
    # set up driver and login in to proquest database
    driver = driver_setup()
    # direct to full text link
    driver.get(full_text_link)

    # from fulltext link to abstract detail link
    make_sure_in_abstract_url(driver)

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("div", class_="display_record_indexing_fieldname")

    ts = soup.find_all("div", class_="display_record_indexing_fieldname")

    titles_text = []
    contents_text = []
    tag1s = soup.find_all("div", class_="display_record_indexing_row")
    for each in tag1s:
        temp_tag = each.find_next("div", class_="display_record_indexing_fieldname")
        title = temp_tag.get_text()
        text = each.get_text()
        content = text.replace(title, "")
        titles_text.append(title)
        contents_text.append(content)

    print(titles_text)
    print(contents_text)
