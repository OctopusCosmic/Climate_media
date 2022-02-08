import pandas as pd


######### process the old version selected and frame into desired format ##########
def get_content(my_location_raw, my_char_start, my_char_end,df): # state, document_number, paragraph_number
    my_location = my_location_raw.split("|")
    state = my_location[0]
    document_number = int(my_location[1])
    paragraph_number = int(my_location[2])
    article_string = df[df["State"]==state].iloc[document_number-1]["article"]
    paragraph = article_string.split("\\")[paragraph_number-1]
    content = paragraph[my_char_start:my_char_end]
    return content

def get_paragraph_length(my_location_raw,df):
    my_location = my_location_raw.split("|")
    state = my_location[0]
    document_number = int(my_location[1])
    paragraph_number = int(my_location[2])
    article_string = df[df["State"]==state].iloc[document_number-1]["article"]
    paragraph = article_string.split("\\")[paragraph_number-1]
    paragraph_length = len(paragraph)
    return paragraph_length


# get the csv from grabing location of each highlighted labeled content in document
#     (Articles like "Alabama Articles.docx")
def process_highlighted_labeled_content(state_abbr,df_f1):
    path = "/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/hand_coded_articles_ready/"
    df_label_a_state = pd.read_csv(f"{path}AL_labeled_content.txt", delimiter = ",", index_col = False)
    # pre-process the data
    df_label_a_state["char_end"] = df_label_a_state["location"].apply(lambda x: get_paragraph_length(x,df_f1))
    df_label_a_state["char_start"] = df_label_a_state["char_start"].fillna(0).astype(int)
    # get the each highlighted labeled content by locations
    df_label_a_state["selected_content"] = df_label_a_state.apply(lambda row: get_content(row.location ,row.char_start ,row.char_end,df_f1), axis=1)

    return df_label_a_state

def main():
    path = "/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/hand_coded_articles_ready/"
    df_f1 = pd.read_csv(f"{path}hand_coded_info_and_full_text_f1.csv", index_col=0)
    df_f2 = pd.read_csv(f"{path}hand_coded_info_and_full_text_f2.csv", index_col=0)
    state_abbr = "AL"
    df_label_a_state = process_highlighted_labeled_content(state_abbr,df_f1)

    # process to desired format
    df_label_a_state["Document number"] = df_label_a_state["location"].apply(lambda x: x.split("|")[1])
    df_label_a_state["labeled frame"] = df_label_a_state["label"].apply(lambda x: x.replace(" ",""))
    df_label_a_state=df_label_a_state.drop(columns=['row', "location","label"])
    df_label_a_state.to_csv(f"{path}AL_selected_content_with_label.csv")

################################################################################

main()

############## RAKE EXAMPLE #################
def main2():
    path = "/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/hand_coded_articles_ready/"
    df_label_a_state = pd.read_csv(f"{path}AL_selected_content_with_label.csv")
    # This csv contains useful information
    # Document number,
    # content: selected content,
    # label: frame










