import pandas as pd

def get_available_states_article(): # cover 49 states
    ### get 50 articles info per state from 49 states web scrape articles
    path = "/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/"
    df_state_names = pd.read_csv(path + "states_abbreviation.csv")
    states = df_state_names["USPS"].tolist()
    # states = ['AL', 'AK', 'AS', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI',
    # 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT',
    # 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC',
    # 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'VI', 'WA', 'WV', 'WI', 'WY']
    states.remove("AK")
    states.remove("AS")
    states.remove("PR")
    states.remove("RI")
    states.remove("VI")
    return states

def preprocessing_concat_to_summary_file(): # Run once !
    # concatenate all article info by states files in to one containing all 49 states
    path = "/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/"

    filename_partial = "new_climate_media_again_"
    states = get_available_states_article()
    df_List = []
    for state in states:
        filename = path + "article_info/" + filename_partial + state + ".csv"
        df_state = pd.read_csv(filename, index_col=False)
        df_List.append(df_state)
    df = pd.concat(df_List,ignore_index=True)
    df.to_csv(path + "/article_info/new_climate_media_again.csv", index_label = 'index')

def main():
    path = "/Users/zhangyuke/PycharmProjects/Climate_Media_beautifulsoup/"
    df = pd.read_csv(path + "article_info/new_climate_media_again.csv")

    # exclude Janel's articles
    # which is stored at hand_coded_info_and_full_text_f1.csv and
    #                    hand_coded_info_and_full_text_f2.csv
    df_exclude1 = pd.read_csv(path + "hand_coded_articles_ready/hand_coded_info_and_full_text_f1.csv")
    df_exclude2 = pd.read_csv(path + "hand_coded_articles_ready/hand_coded_info_and_full_text_f2.csv")

    s1 = df.shape
    df = df[~(df["Title"].isin(df_exclude1["Title"]) | df["Title"].isin(df_exclude2["Title"]))]
    s2 = df.shape
    diff = s1[0] - s2[0]
    print("The number of articles Janel hand coded is " + str(diff))

    states = get_available_states_article()
    size_per_state = df.groupby("State").size()
    print(f'size_per_state: {size_per_state}')


    states_lt50 = size_per_state[size_per_state<=50].index
    states_gt50 = size_per_state[size_per_state>50].index
    print(f'states_truncated: {states_lt50}')
    print(f'states_after_truncated: {states_gt50}')

    df_gt50 = df[df["State"].isin(states_gt50)] # articles from states having greater than 50 articles
    df_lt50 = df[df["State"].isin(states_lt50)] # articles from states having greater than 50 articles
    df_lt50 = df_lt50.drop(columns=['index', 'previous_index'])

    sample_df_gt50 = df_gt50.groupby("State").sample(n=50, random_state=1).reset_index(drop=True)
    sample_df_gt50 = sample_df_gt50.drop(columns=['index', 'previous_index'])

    sample_df = pd.concat([df_lt50,sample_df_gt50]).reset_index(drop=True)

    sample_df.to_csv(path+"/article_info/new_climate_media_again_sample_by_state.csv", index_label = 'index')


main()