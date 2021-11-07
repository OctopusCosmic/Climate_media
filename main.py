import docx2txt

#from docx import Document
#def main():
    #filename = "Alaska Articles.docx"
    #document = Document(filename)
    #count = 0
    #for para in document.paragraphs:
        #count += 1
import pandas as pd


def docx_to_txt(statename):
    # Passing docx file to process function
    path = "/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/hand_coded_articles/"
    filename = path + statename + " Articles.docx"
    text = docx2txt.process(filename)

    # Saving content inside docx file into output.txt file
    with open(f"{path}{statename} Articles.txt", "w") as text_file:
        print(text, file=text_file)
def convert_all_docx_to_txt():
    statenames = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
                  "Connecticut", "Delware", "Florida", "Georgia"]

    path = "/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/hand_coded_articles/"
    # only run once
    for i in range(len(statenames)):
        if i == 0:
            statename = statenames[i]
            docx_to_txt(statename)




def format2_get_links():
    path = "/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/hand_coded_articles/"
    statenames2 = ["Arizona", "California", "Connecticut", "Delware", "Florida", "Georgia"]
    statenames2_initial = ["AZ", "CA", "CT", "DE", "FL", "GA"]
    for i in range(2, len(statenames2)):
        statename = statenames2[i]
        statename_initial = statenames2_initial[i]

        with open(f"{path}{statename} Articles.txt", "r") as text_file:
            header = text_file.readline().rstrip("\n")
            # print(f"{path}{statename} Articles.txt")
            print(header)

            # Formart 2 (Proquest-> we need Document url):
            # Arizona/California/Connecticut/Delaware/Florida/Georgia
            line = text_file.readline().rstrip("\n")
            article_links = []
            x = True
            while x:
                line = text_file.readline()
                if line != "":
                    # get all links
                    if line[:5] == "https":
                        article_link = line.rstrip("\n")
                        article_links.append(article_link)

                if not line:
                    print('EOF')
                    x = False

        df = pd.DataFrame(columns=["State", "article_link"])
        df["article_link"] = article_links
        df["State"] = statename_initial
        df.to_csv(f"{path}{statename}_article_links.csv")


def put_article_links_together():
    path = "/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/hand_coded_articles/"
    statenames2 = ["Arizona", "California", "Connecticut", "Delware", "Florida", "Georgia"]
    df_list =[]
    for i in range(len(statenames2)):
        statename = statenames2[i]
        filename = f"{path}{statename}_article_links.csv"
        df = pd.read_csv(filename, index_col=0)
        df_list.append(df)

    df_whole = pd.concat(df_list)
    df_whole.reset_index(inplace=True, drop=True)
    df_whole.to_csv(f"{path}hand-coded_article_links.csv")

#put_article_links_together() -> storeed in hand-coded_article_links.csv


def main():
    statenames = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado",
                  "Connecticut","Delware","Florida","Georgia"]

    path = "/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/hand_coded_articles/"
    statenames1 = ["Alabama","Alaska","Arkansas","Colorado"]
    statenames2 = ["Arizona","California","Connecticut","Delware","Florida","Georgia"]
    statenames2_initial  = ["AZ","CA","CT","DE","FL","GA"]

    statename = statenames1[0]
    with open(f"{path}{statename} Articles.txt", "r") as text_file:
        header = text_file.readline().rstrip("\n")
        #print(f"{path}{statename} Articles.txt")
        print(header)

        # Format 1: Alabama/Alaska/Arkansas/Colorado
        '''
        line = text_file.readline().rstrip("\n")
        article_count = 0
        paragraph_count = 0
        document_list = []
        articles = []
        titles = []
        x = True
        while x:
            line = text_file.readline()
            if line != "":
                if line[:8] == "Document":
                    # print(line)
                    article_count += 1
                    print(article_count)
                    # print(paragraph_count)
                    paragraph_count = 0
                    if len(document_list) != 0:

                        title = document_list[0].rstrip("\n")
                        article = "\\".join(document_list)
                        #if document_list[1][:2] == "By":
                            #print(f"Author: {document_list[1][3:]}")
                        articles.append(article)
                        titles.append(title)

                    #print(document_list)
                    document_list = []

                else:  # line[:8] != "Document"
                    paragraph_count += 1
                    if line != "\n":
                        document_list.append(line)

            if not line:
                print('EOF')
                x = False
                title = document_list[0].rstrip("\n")
                article = "\\".join(document_list)
                articles.append(article)
                titles.append(title)
        '''
    

