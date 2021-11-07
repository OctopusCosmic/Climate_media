### Climate_media_research by Yuke Zhang(Purdue University)
### Last Updated Date: Oct 9th, 2021

import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

pd.set_option('display.max_columns', None)

def get_title(): #not important
    df = pd.DataFrame(columns=["Keyword", "Title", "Author", "Publication_title", "Publication_date",
                               "Place_of_publication", "First_page", "Section", "Document_URL"])
    newObs = ["Keyword", "Title", "Author", "Publication_title", "Publication_date",
              "Place_of_publication", "First_page", "Section", "Document_URL"]  ##
    idx = list(df.columns)  ##
    newExample = pd.Series(newObs, name=0, index=idx)  ##
    df = df.append(newExample)  ##
    df.to_csv("test.csv")


def login_to_proquest(driver,web_link):
    duo = input("Password HERE is: ")
    # get access to webpage
    driver.get(web_link)

    # login to boiler account
    username = "" # input your username here
    password = "" + duo  # input your password here
    driver.find_element_by_name("username").send_keys(username)
    driver.find_element_by_name("password").send_keys(password)
    driver.find_element_by_name("submit").click()

    after_login_url = driver.current_url
    return after_login_url

def make_sure_in_full_text_url(driver):
    html1 = driver.page_source
    soup1 = BeautifulSoup(html1, "html.parser ")
    # we assume the full text tab is the first tag we find
    tag = soup1.find("li", id="tab-Fulltext-null")
    # if given attribute is false
    if tag["aria-selected"] == "false":
        driver.find_element_by_id("tab-Fulltext-null").click()


def Full_text_scraping(driver, url):
    # enter the document webpage
    driver.get(url)
    time.sleep(2)

    make_sure_in_full_text_url(driver)

    # we need to update soup
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    tag_display_record_section = soup.find("div", class_ = "display_record_section_copy")

    tag_display_record_text = tag_display_record_section.contents[0]
    tag_fulltext_or_textgraphic = tag_display_record_text.contents[0]

    if (tag_fulltext_or_textgraphic.contents[0]).name == "root": # text with graphic
        root = tag_fulltext_or_textgraphic.contents[0]
        tag_wordcount = root.contents[0]
    else: # full text
        tag_wordcount = tag_fulltext_or_textgraphic.contents[1]

    article = [] #each paragraph as an element
    count = 0
    for child in tag_wordcount.children:
        count += 1
        try:
            if child.name == "p":
                paragraph = child.text
                article.append(paragraph)
                print(paragraph)
        except:
            continue

    return article

## currently working on Jupyter Notebook

def driver_setup():
    # set up chrome driver
    driver = webdriver.Chrome("/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/chromedriver_v94")
    driver.set_page_load_timeout(60)
    web_link = "https://www.proquest.com/usnews/advanced?accountid=13360"
    #login_to_proquest(driver, web_link) # without login
    after_login_url = web_link
    return driver

def add_info(row, df, article):
    newObs = ["NA"]
    idx = list(df.columns)
    # add this article information to a row of dataframe
    newObs = [article]

    newExample = pd.Series(newObs, name=row, index=idx)
    df = df.append(newExample)

    return df

def adjust_storing_article_format(article):
    return "\\".join(article)

## function about filter by state
def standardize_format(place):
    my_dictionary = {"Arlington": "Arlington, Tex.",
                     "Austin": "Austin, Tex.",
                     "Los Angeles": "Los Angeles, Calif.",
                     "Fort Lauderdale": "Fort Lauderdale, Fla.",
                     "Denver": "Denver, Colo.",
                     "Annapolis": "Annapolis, Md.",
                     "Philadelphia": "Philadelphia, Pa.",
                     "Boston": "Boston, Mass.",
                     "Tampa Bay": "Tampa Bay, Fla.",
                     "Charleston, W.V.": "Charleston, W.Va.",
                     "Hutchinson, Kan": "Hutchinson, Kan.",
                     "Toms River, N. J.": "Toms River, N.J.",
                     "Bismarck, ND": "Bismarck, N.D.",
                     "Jacksonville, NC": "Jacksonville, N.C.",
                     "Kinston, NC": "Kinston, N.C.",
                     "Havelock, NC": "Havelock, N.C.",
                     "New Bern, NC": "New Bern, N.C.",
                     "New York": "New York, N.Y.",
                     "Janesville, WI": "Janesville, Wis.",
                     "Albuquerque": "Albuquerque, N.M."}
    for key in my_dictionary.keys():
        if place == key:
            place = my_dictionary[key]
    return place

def abbreviate(x):
    df_abbr = pd.read_csv("states_abbreviation.csv")
    state_abbr = ""
    # two conditions
    try:
        # 1. have two words separated by a comma and a space
        alist = x.split(", ")
        state_abbr = alist[1]
        # 2. the traditional abbreviation of state name is in US 50 states
        state_abbr = df_abbr[df_abbr["Traditional*"] == state_abbr]["USPS"].iloc[0]
    except:
        # those does not satisfy both conditions retain
        state_abbr = x # México City and Guam
        #print(x)
    return state_abbr

def filter_by_state(df): # df: new_climate_media.csv
    # apply standarize_format to get rid of special case, having most of them matching states traditional abbreviation
    df['Place_of_publication'] = df['Place_of_publication'].apply(lambda x: standardize_format(x))
    # apply abbreviate to find out the abbreviation of states for each place
    df['State'] = df['Place_of_publication'].apply(lambda x: abbreviate(x))
    # trim cases like México City and Guam, keep every place in US
    df = df[df['State'].str.len() == 2]
    # alphabetically states
    states = sorted(df["State"].unique())

    return states, df

def divide_data_by_state():
    # read document url from my dataset new_climate_media.csv
    df = pd.read_csv("article_info/new_climate_media_again.csv")
    df = df.drop("Unnamed: 0", axis=1)

    # add a row which is abbreviation of state for each article
    states, df = filter_by_state(df)

    # divide new_climate_media_again by states
    for current_state in states:
        df_a_state = df[df["State"] == current_state]
        df_a_state.to_csv(f"new_climate_media_again_{current_state}.csv")


def main_by_state(current_state):
    # read document url from my dataset new_climate_media.csv
    df_a_state = pd.read_csv(f"new_climate_media_again_{current_state}.csv")
    df_a_state = df_a_state.drop("Unnamed: 0", axis=1)

    full_text_urls = df_a_state["Document_URL"].tolist()

    # initialize value
    row = 1540
    gap = 20  # Since a duo code only valid for 40 articles, further the account has been logged out
    low, high = row, row + gap
    length = len(df_a_state)
    print("Total length is: "+ str(length))

    while (row < length):
        df1 = pd.DataFrame(columns=["article"])
        # login every 40 articles
        driver = driver_setup()

        while (low <= row < high and row < length):
            url = full_text_urls[row]
            print(row)
            print(url)
            article = Full_text_scraping(driver, url)
            print(article)
            article = adjust_storing_article_format(article)
            #articles.append(article)
            df1 = add_info(row, df1, article)
            # update row
            row += 1

        print("row: " + str(row) + " (" + str(low) + ", " + str(high) + ")")

        # append info of 40 articles to the file
        df1.to_csv(f"climate_media_full_text_{current_state}.csv", mode='a', header=False)

        # update lower and upper bound
        low += gap
        high += gap
        # quit current driver
        driver.quit()


def test():
    driver = driver_setup()
    url = "https://www.proquest.com/docview/268298662?accountid=13360"
    Full_text_scraping(driver, url)

#test()
main_by_state("NY")


# there are
# CA(799) pdf get it manually please
#

