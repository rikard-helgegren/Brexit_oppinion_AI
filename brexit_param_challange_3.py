import re
import numpy as np
import sys

from sklearn.datasets import load_files

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.svm          import LinearSVC
from sklearn.linear_model import LogisticRegression  
from sklearn.linear_model import Perceptron

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


from sklearn.metrics import accuracy_score

#from sklearn.model import coef_

def clean_Up_String(sentence):
    #sentence = sentence.rstrip()
    sentence = sentence.replace(u'\xa0', u' ')
    sentence = sentence.replace(u'\ufeff', u' ')
    sentence = sentence.replace(u'\u200b', u' ')
    sentence = sentence.replace(u'...', u' ')
    sentence = sentence.replace(".", "")
    sentence = sentence.replace(",", "")
    sentence = sentence.replace("!", "")
    sentence = sentence.replace("?", "")
    sentence = sentence.replace("(", "")
    sentence = sentence.replace(")", "")
    sentence = sentence.replace("\\", "")
    sentence = sentence.replace('\"', "")
    sentence = sentence.replace("\'", "")
    sentence = sentence.replace("£", "")
    sentence = sentence.replace(":", "")
    sentence = sentence.replace("-", "")
    sentence = sentence.replace("’", "")
    sentence = sentence.replace("'", "")
    sentence = sentence.replace("%", " %")
    sentence = sentence.lower()
    sentence = sentence.rstrip()

    return sentence

def clean_Up_Lables(lables):
    return_lable = []
    for i in lables:
        return_lable = i[0]
    return return_lable

def read_feature_descriptions(filename):
    valOpinion = []
    label_of_sentence = []
    with open(filename) as f:
        for l in f:
            colums = re.split(r'\t+', l)
            label_of_sentence.append(clean_Up_Lables(colums[0]))
            sentence = clean_Up_String(colums[-1])
            valOpinion.append(sentence)
    return valOpinion, label_of_sentence


#feat_valOpinion, feat_comment = read_feature_descriptions('datasets/tryBrexitData')
#feat_valOpinion, feat_comment = read_feature_descriptions('datasets/givenBrexitData1')
feat_valOpinion, feat_comment = read_feature_descriptions('datasets/fullBrexitData')
#print(feat_comment, "\n\n", feat_valOpinion)

# split the dataset in training and test set:
#42
#print("Train data:\n\n", docs_train, "\n\nTrain answer:\n\n", y_train)

"""train_test_split: Split arrays or matrices into random train and test subsets. Quick utility that wraps input validation and next(ShuffleSplit().split(X, y)) and application to input data into a single call for splitting (and optionally subsampling) data in a oneliner."""

def run():
    docs_train, docs_test, y_train, y_test = train_test_split(
                                             feat_valOpinion, feat_comment, test_size=0.25, random_state=None)

    global docs_train, docs_test, y_train, y_test, feat_valOpinion, feat_comment
    vect = TfidfVectorizer(max_df = 0.5)
    clf = LogisticRegression(C = 2)
    #clf = LogisticRegression()


    pipeline = make_pipeline(
        vect,
        clf #C=1000 Penalty parameter
        )

    pipeline.fit(docs_train, y_train) #-added-


    y_guess = pipeline.predict(docs_test)
    Score_1 = accuracy_score(y_test, y_guess)

    ####################
    Tfidf_min_df = 2
    Tfidf_max_df = 0.3
    Tfidf_max_features = None # None 10000 500000 1000000


    LR_C =0.8 
    LR_Penalty = "l2" #penalty : str, ‘l1’ or ‘l2
    LR_max_iter = 1

    clf = LogisticRegression(C = 2, )
    vect = TfidfVectorizer(max_df = 0.5)

    pipeline = make_pipeline(
        vect,
        clf #C=1000 Penalty parameter
        )

    pipeline.fit(docs_train, y_train) #-added-


    y_guess = pipeline.predict(docs_test)
    Score_2 = accuracy_score(y_test, y_guess)


    if Score_1 > Score_2:
        print(Score_1, "Prev better", Score_2)
    elif Score_1 < Score_2:
        print(Score_1, "New better", Score_2)
    else:
        print(Score_1, "Draw", Score_2)



for i in range(1,100):
    run()


