import sys
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
    paragraph_number = int(my_location[2] )
    article_string = df[df["State"]==state].iloc[document_number-1]["article"]
    paragraph = article_string.split("\\")[paragraph_number-1]
    paragraph_length = len(paragraph)
    return paragraph_length


# get the csv from grabing location of each highlighted labeled content in document
#     (Articles like "Alabama Articles.docx")
def process_highlighted_labeled_content(state_abbr,df_f1):
    path = "hand_coded_articles_ready/"
    df_label_a_state = pd.read_csv(f"{path}AL_labeled_content.txt", delimiter = ",", index_col = False)
    # pre-process the data
    df_label_a_state["char_end"] = df_label_a_state["location"].apply(lambda x: get_paragraph_length(x,df_f1))
    df_label_a_state["char_start"] = df_label_a_state["char_start"].fillna(0).astype(int)
    # get the each highlighted labeled content by locations
    df_label_a_state["selected_content"] = df_label_a_state.apply(lambda row: get_content(row.location ,row.char_start ,row.char_end,df_f1), axis=1)

    return df_label_a_state

def main():
    path = "hand_coded_articles_ready/"
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







############## RAKE EXAMPLE #################
import operator
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


#for text pre-processing
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords') # download once is good
# nltk.download('punkt') # download once is good
# nltk.download('averaged_perceptron_tagger') # download once is good
# nltk.download('wordnet') # download once is good
# nltk.download('omw-1.4') # download once is good

snow = SnowballStemmer('english')
wl = WordNetLemmatizer()
pd.set_option("display.max_columns", None)

#for bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#for word embedding
import gensim
from gensim.models import Word2Vec #Word2Vec is mostly used for huge datasets

#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

def visualization(df,state_abbr,tasks):
    path = "hand_coded_articles_ready/"
    # draw number of observations for each frame in AL file
    x = df["labeled frame"].value_counts()
    print(x)
    plt.bar(x.index, x)
    plt.xticks(rotation = 30)
    plt.title("the number of each labeled frame")
    plt.show()
    plt.savefig(f"{path}{state_abbr}_each_labeled_frame_number.jpg")

    print(df["labeled frame"].unique())
    labeled_frames = ['F2-L-P', 'F1-L-P', 'F1-P', 'F2-P', 'F1-N', 'S1-N', 'F2-N', 'E3-N', 'E1-P', 'S2-N', 'S2-P',
                      'S1-P', 'F2-L-N', 'E1-L-P', 'E3-L-P', 'E2-P', 'E3-P', 'E1-N']
    for task in tasks:
        # draw word_count frequency for each frame in AL file
        fig, axs = plt.subplots(6, 3, figsize=(9, 18))
        for i in range(6):
            for j in range(3):
                train_words = df[df['labeled frame'] == labeled_frames[3*i+j]][task]
                axs[i,j].hist(train_words)
                axs[i,j].set_title(labeled_frames[3*i+j]+" "+task)
        for ax in axs.flat:
            ax.set(xlabel = task, ylabel='frequency')
        for ax in axs.flat:
            ax.label_outer()
        plt.show()
        plt.savefig(f"{path}{state_abbr}_each_labeled_frame_{task}_frequency.jpg")



# convert to lowercase and remove punctuations and characters and then strip
def preprocess(text):
    text = text.lower()  # lowercase text
    text = text.strip()  # get rid of leading/trailing whitespace
    text = re.compile('<.*?>').sub('', text)  # Remove HTML tags/markups
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ',
                                                                  text)  # Replace punctuation with space. Careful since punctuation can sometime be useful
    text = re.sub('\s+', ' ', text)  # Remove extra space and tabs
    text = re.sub(r'\[[0-9]*\]', ' ', text)  # [0-9] matches any digit (0 to 10000...)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)  # matches any digit from 0 to 100000..., \D matches non-digits
    text = re.sub(r'\s+', ' ',
                  text)  # \s matches any whitespace, \s+ matches multiple whitespace, \S matches non-whitespace

    return text

def calcuate_text_basic_count(df):
    # 1. WORD-COUNT
    df['word_count'] = df['selected_content'].apply(lambda x: len(str(x).split()))
    # 2. CHARACTER-COUNT
    df['char_count'] = df['selected_content'].apply(lambda x: len(str(x)))
    # 3. UNIQUE WORD-COUNT
    df['unique_word_count'] = df['selected_content'].apply(lambda x: len(set(str(x).split())))

    return df

def stopword_removal(string): #1. STOPWORD REMOVAL
    a = [i for i in string.split() if i not in stopwords.words('english')]
    #print(f"stop words:{stopwords.words('english')}")
    return ' '.join(a)

def stemming(string):
    #print(f"tocken:{word_tokenize(string)}")
    a=[snow.stem(i) for i in word_tokenize(string) ]
    return " ".join(a)

def lemmatizer(string): # Tokenize the sentence
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    #print(f"word position tags:{word_pos_tags}")
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def finalpreprocess(string):
    # Preprocess data/# 1. STOPWORD REMOVAL/# 3. LEMMATIZATION
    return lemmatizer(stopword_removal(preprocess(str(string))))

def show_text_preprocessing(string):
    # Preprocess data
    text = preprocess(str(string))
    print(f"preprocess_text={text}")

    #### nltk preprocess  ####/*
    # 1. STOPWORD REMOVAL
    text = stopword_removal(text)
    print(f"stopword_removal_text={text}")

    # 2. STEMMING
    text = stemming(text)
    print(f"stemming_text={text}")

    # 3. LEMMATIZATION
    text = lemmatizer(text)
    print(f"lemmatizer_text={text}")

def transform(X, w2v): # X is candidate words list
    dim = len(next(iter(w2v.values())))
    result = np.array([
        np.mean(
            [w2v[w] for w in words if w in w2v]
            or [np.zeros(dim)], axis=0
        ) for words in X
    ]) # find the mean of each word vector and store them into a list
    return result

def new_main():
    path = "hand_coded_articles_ready/"
    state_abbr = "AL"
    filename = f"{path}{state_abbr}_selected_content_with_label.csv"
    df_label_a_state = pd.read_csv(filename,index_col=0)
    df_label_a_state = df_label_a_state.dropna()

    ### download needed package
    # nltk.download('stopwords') # download once is good
    # nltk.download('punkt') # download once is good
    # nltk.download('averaged_perceptron_tagger') # download once is good
    # nltk.download('wordnet') # download once is good
    # nltk.download('omw-1.4') # download once is good

    ### calculate word_count, char_count, unique_word_count for initial data
    #df_label_a_state = calcuate_text_basic_count(df_label_a_state)
    # tasks = ['word_count', 'char_count', 'unique_word_count']
    # visualization(df_label_a_state, "AL", tasks)

    ### show the preprocess steps for a text
    # ori_text = df_label_a_state["selected_content"].iloc[0]
    # print(f"text={ori_text}")
    # show_text_preprocessing(ori_text)

    ### apply preprocess data, STOPWORD REMOVAL, and LEMMATIZATION to selected content
    df_label_a_state['clean_text'] = df_label_a_state["selected_content"].apply(lambda x: finalpreprocess(x))
    #df_label_a_state.to_csv(f"{path}{state_abbr}_selected_content_with_label.csv")
    df_label_a_state = df_label_a_state.dropna()
    df_label_a_state.to_csv(filename)


def onehot(voca, values): # voca: list of vocabulary, values: list of test values
    encoded_values = [i for e in values for i in range(len(voca)) if voca[i] == e]
    '''    #expanded nested loop is shown below
        for e in values:
            for i in range(len(voca)):
                if voca[i] == e:
                    print(i) '''
    return np.eye(len(voca))[encoded_values]



import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torchsummary import summary
import torch.nn.functional as functional
def onehotback(tensor):
    matrix = tensor.numpy()
    new_matrix = [j for tuple in matrix for j in range(len(tuple)) if tuple[j] == 1]
    '''new_matrix = []
        for tuple in matrix:
            for j in range(len(tuple)):
                if tuple[j] == 1:
                    new = j
                    new_matrix.append(new)'''
    return torch.tensor(new_matrix)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(100, 80),  # 80 is size of hidden layer
            nn.ReLU(),
            nn.Linear(80, 18),  # tuning this
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def main2():
    path = "hand_coded_articles_ready/"
    state_abbr = "AL"
    # This csv contains useful information
    # Document number, selected content, labeled frame

    filename = f"{path}{state_abbr}_selected_content_with_label.csv"
    df_label_a_state = pd.read_csv(filename, index_col=0)
    #print(df_label_a_state.head())
    df_label_a_state = df_label_a_state.dropna()

    # Create word2vec
    df_label_a_state['clean_text_token'] = [nltk.word_tokenize(i) for i in df_label_a_state['clean_text']]  # convert preprocessed sentence to tokenized sentence
    model = Word2Vec(df_label_a_state['clean_text_token'], min_count=1, vector_size=100) ##min_count=1 means word should be present at least across all documents
    w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))  # combination of word and its vector

    # Prepare data with train and test
    X_train, X_val, y_train, y_val = train_test_split(df_label_a_state["clean_text"],
                                                      df_label_a_state["labeled frame"],
                                                      test_size=0.2,
                                                      shuffle=True)
    X_train_tok = [nltk.word_tokenize(i) for i in X_train]  # for word2vec
    X_val_tok = [nltk.word_tokenize(i) for i in X_val]  # for word2vec
    y_train = y_train.to_list()
    y_val = y_val.to_list()
    frame_voca = ['F2-L-P', 'F1-L-P', 'F1-P', 'F2-P', 'F1-N', 'S1-N', 'F2-N', 'E3-N', 'E1-P', 'S2-N', 'S2-P', 'S1-P', 'F2-L-N', 'E1-L-P', 'E3-L-P', 'E2-P', 'E3-P', 'E1-N']
    y_train_vec = onehot(frame_voca, y_train)
    y_val_vec = onehot(frame_voca, y_val)
    # Word2Vec
    X_train_vectors_w2v = transform(X_train_tok, w2v)
    X_val_vectors_w2v = transform(X_val_tok, w2v)

    print(f"y_train_vec.shape={len(y_train_vec)}")
    print(f"y_val_vec.shape={len(y_val_vec)}")
    print(f"X_train_vectors_w2v={X_train_vectors_w2v.shape}")
    print(f"X_val_vectors_w2v={X_val_vectors_w2v.shape}")
    print("y_train")
    print(y_train)
    print("y_val")
    print(y_val)

    X_train_vectors = torch.tensor(X_train_vectors_w2v)
    X_val_vectors = torch.tensor(X_val_vectors_w2v)
    y_train_vec = torch.tensor(y_train_vec)
    y_val_vec = torch.tensor(y_val_vec)


    epochs = 450
    batch_size_original = 64
    learning_rate = 0.01
    nnet = Net()
    grad_desc = optim.SGD(nnet.parameters(), lr=learning_rate)
    dimension = 100
    loss_func = nn.CrossEntropyLoss()
    for e in range(epochs):
        size = batch_size_original
        for i in range(0, len(X_train_vectors), size):
            if i + size > len(X_train_vectors):
                size = len(X_train_vectors) - i
            grad_desc.zero_grad()
            data = autograd.Variable(X_train_vectors[i:(i + size)].data.view(size, dimension), requires_grad=True)
            predictions = nnet(data)
            labels = autograd.Variable(y_train_vec[i:(i + size)].data)
            loss = loss_func(predictions, labels)
            loss.backward(retain_graph=True)
            grad_desc.step()

    #pred = functional.log_softmax(nnet(X_val_vectors), dim=1)
    #pred = torch.max(nnet(X_val_vectors).data, 1)
    logits = nnet(X_val_vectors.data)
    pred = logits.max(1).indices
    y_val_result = onehotback(y_val_vec)
    print(pred)
    print(y_val_result)
    acc = (pred == onehotback(y_val_vec)).sum().item() / pred.size(0)
    print(acc)



    '''
    # TF-IDF
    # Convert x_train to vector since model can only run on numbers and not words- Fit and transform
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train)  # tfidf runs on non-tokenized sentences unlike word2vec
    # Only transform x_test (not fit and transform)
    X_val_vectors_tfidf = tfidf_vectorizer.transform(X_val)  # Don't fit() your TfidfVectorizer to your test data: it will
    # change the word-indexes & weights to match test data. Rather, fit on the training data, then use the same train-data-
    # fit model on the test data, to reflect the fact you're analyzing the test data only based on what was learned without
    # it, and the have compatible

    
    print("Logistic Regression(tf-idf)")
    lr_tfidf = LogisticRegression(solver='liblinear', C=10, penalty='l2')
    lr_tfidf.fit(X_train_vectors_tfidf, y_train)  # model

    # Predict y value for test dataset
    y_predict = lr_tfidf.predict(X_val_vectors_tfidf)
    y_prob = lr_tfidf.predict_proba(X_val_vectors_tfidf)[:, 1]

    print(classification_report(y_val, y_predict))
    print('Confusion Matrix:', confusion_matrix(y_val, y_predict))

    print("Naive Bayes(tf-idf)")
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_vectors_tfidf, y_train)  # model

    # Predict y value for test dataset
    y_predict = nb_tfidf.predict(X_val_vectors_tfidf)
    y_prob = nb_tfidf.predict_proba(X_val_vectors_tfidf)[:, 1]

    print(classification_report(y_val, y_predict))
    print('Confusion Matrix:', confusion_matrix(y_val, y_predict))

    print("Logistic Regression (W2v)")
    # FITTING THE CLASSIFICATION MODEL using Logistic Regression (W2v)
    lr_w2v = LogisticRegression(solver='liblinear', C=10, penalty='l2')
    lr_w2v.fit(X_train_vectors_w2v, y_train)  # model

    # Predict y value for test dataset
    y_predict = lr_w2v.predict(X_val_vectors_w2v)
    y_prob = lr_w2v.predict_proba(X_val_vectors_w2v)[:, 1]

    print(classification_report(y_val, y_predict))
    print('Confusion Matrix:', confusion_matrix(y_val, y_predict))
    '''
main2()















