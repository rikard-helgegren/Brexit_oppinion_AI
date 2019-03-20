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

import time
start_time = time.time()

#from sklearn.model import coef_

def clean_Up_String(sentence):
    #sentence = sentence.rstrip()
    sentence = sentence.replace(u'\xa0', u' ')
    sentence = sentence.replace(u'\ufeff', u' ')
    sentence = sentence.replace(u'\u200b', u' ')
    sentence = sentence.replace(u'...', u' ')
    sentence = sentence.replace(u'…', u' ')
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


def get_word_with_score(scores, all_scores, all_words):
    score_words = []
    for score in scores:
      ids = [i for i,x in enumerate(all_scores) if x == score]
      for idx in ids: 
         score_words.append(score)
         score_words.append(all_words[idx])


    return score_words



def run_program():
    global clf, vect
    #feat_valOpinion, feat_comment = read_feature_descriptions('datasets/tryBrexitData')
    feat_valOpinion, feat_comment = read_feature_descriptions('datasets/givenBrexitData1')
    #print(feat_comment, "\n\n", feat_valOpinion)

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
    feat_valOpinion, feat_comment, test_size=Split_test_size, random_state=Split_random_state)#42



    #vect = TfidfVectorizer(min_df=TfidfV_min_df, max_df=Tfidf_max_df)
    #vect = CountVectorizer()
    #vect = HashingVectorizer()

    #clf = LinearSVC()
    #clf = Perceptron()
    #clf = LogisticRegression()


    pipeline = make_pipeline(
        vect,
        clf 
        )

    pipeline.fit(docs_train, y_train) #-added-

    y_guess = pipeline.predict(docs_test)
    score.append(accuracy_score(y_test, y_guess))





TfidfV_min_df = 2
Tfidf_max_df = 0.95

Split_test_size= 0.25
Split_random_state = np.random.randint(2000)#42, 22, 1765, 1619

score = []

maxScore = 0
bestParams = []

clf = LinearSVC()
vect = TfidfVectorizer(min_df=TfidfV_min_df, max_df=Tfidf_max_df)
v = 0
m = 0


def varry_Vec():
    global v, vect
    for v in range(0,3):
        if v == 0:
            vect = TfidfVectorizer()#min_df=TfidfV_min_df, max_df=Tfidf_max_df)
        elif v == 1:
            vect = CountVectorizer()#min_df=TfidfV_min_df, max_df=Tfidf_max_df)
        elif v == 2:
            vect = HashingVectorizer()
        varry_model()

def varry_model():
    global m, clf
    for m in range(0,3):
        if m == 0:
            clf = LinearSVC()
        elif m == 1:
            clf = Perceptron()
        elif m == 2:
            clf = LogisticRegression()
        itter_times()

def itter_times():
    global v, m, score, maxScore, bestParams
    for i in range(0,1):
        #Split_random_state = np.random.randint(2000)#42 , 22

        run_program()

        #print("random_state: ", Split_random_state)
        #print("v, m:", v, m)
        #print("\nScore: ", score[-1])
        #print("\n########################\n")
        if score[-1] > maxScore:
            maxScore = score[-1]
            bestParams = [v, m, Split_random_state]     

for p in range(0,1000):
    varry_Vec()           

    Split_random_state = np.random.randint(2000)  
    print("best params: ", bestParams)
    maxScore = 0
    bestParams = 0


#print("\nMax: ",maxScore)
print("\n--- %s seconds ---" % (time.time() - start_time))