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

def clean_Up_String(sentence):
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


docs_train, y_train = read_feature_descriptions('Filpath_To_Trainingdata')
docs_test, y_test = read_feature_descriptions('Filpath_To_Testingdata')


vect = TfidfVectorizer(max_df=0.5)
clf  = LogisticRegression(C=2)


pipeline = make_pipeline(
    vect,
    clf 
    )

pipeline.fit(docs_train, y_train) 


y_guess = pipeline.predict(docs_test)
print("\nScore: ",accuracy_score(y_test, y_guess))



    #------------------Fun time---------------------#
    #sorted_score = sorted(clf.coef_[0])
    #top_scores = list(reversed(sorted_score))[:10]

    #print("\n-----Pro WORDS---------\n", get_word_with_score(top_scores, clf.coef_[0], vect.get_feature_names()))
    #print("\n-----Against WORDS---------\n",get_word_with_score(sorted_score[:10], clf.coef_[0], vect.get_feature_names()))
    #print("\n########################\n")
