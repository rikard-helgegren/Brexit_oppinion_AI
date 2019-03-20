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

def read_feature_descriptions(filename):
    valOpinion = []
    comments = []
    with open(filename) as f:
        for l in f:
            colums = re.split(r'\t+', l)
            comments.append(colums[0])
            sentence = clean_Up_String(colums[-1])
            valOpinion.append(sentence)
    return valOpinion, comments



def get_word_with_score(scores, all_scores, all_words):
    score_words = []
    for score in scores:
      ids = [i for i,x in enumerate(all_scores) if x == score]
      for idx in ids: 
         score_words.append(score)
         score_words.append(all_words[idx])


    return score_words



def run_program():
    global clf, vect#, Tfidf_min_df, Tfidf_max_df, Tfidf_max_features, LR_C, LR_Penalty, LR_max_iter
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
        clf #C=1000 Penalty parameter
        )

    pipeline.fit(docs_train, y_train) #-added-

    y_guess = pipeline.predict(docs_test)
    score.append(accuracy_score(y_test, y_guess))
    #print("\nScore: ",accuracy_score(y_test, y_guess))

    #------------------Fun time---------------------#
    #sorted_score = sorted(clf.coef_[0])
    #top_scores = list(reversed(sorted_score))[:10]

    #print("\n-----Pro WORDS---------\n", get_word_with_score(top_scores, clf.coef_[0], vect.get_feature_names()))
    #print("\n-----Against WORDS---------\n",get_word_with_score(sorted_score[:10], clf.coef_[0], vect.get_feature_names()))
    #print("\n########################\n")




Tfidf_min_df = 2
Tfidf_max_df = 0.3
Tfidf_max_features = None # None 10000 500000 1000000


LR_C =0.8 
LR_Penalty = "l2"
LR_max_iter = 1

Split_test_size= 0.25
Split_random_state = np.random.randint(2000)#42, 22, 1765, 1619


score = []

maxScore = 0
bestParams = []

clf = LogisticRegression(penalty = LR_Penalty, C = LR_C, max_iter=LR_max_iter)
vect = TfidfVectorizer(min_df=Tfidf_min_df, max_df=Tfidf_max_df, max_features = Tfidf_max_features )
v = 0
m = 0


def varry_Vec():
    global v, vect, Tfidf_min_df, Tfidf_max_df, Tfidf_max_features, LR_C, LR_Penalty, LR_max_iter
    for min_df in range(2,3):#range(0,4):
        #print("1")
        #print(min_df)
        for i in range(1,6):#range(1,20):
            #print("2")
            Tfidf_max_df = round(0.2 + 0.05*(i), 3)
            for j in range(0,1):#range(0,5):
               # print("3")
                if v == 0:
                    Tfidf_max_features = None
                else :
                    Tfidf_max_features = 100*(10**j)
                vect = TfidfVectorizer(min_df=Tfidf_min_df, max_df=Tfidf_max_df, max_features = Tfidf_max_features )
                varry_model()

def varry_model():
    global m, clf, LR_C, LR_Penalty, LR_max_iter, Tfidf_min_df, Tfidf_max_df, Tfidf_max_features
    for c in range(0,1):#range(1,10):
        
        #print("4")
        if c ==0:
            LR_C = 0.8
        elif c <=7:
            LR_C = c/5
        elif c > 5:
            LR_C = (c-7)
        #print(LR_C)
        for p in range(1,2):
            #print("5")
            if p == 0:
                LR_Penalty = 'l1'
            elif p == 1:
               LR_Penalty = 'l2'
            for itter in range(1,2):#range(0,10):
                #print("6")
                LR_max_iter = itter
                clf = LogisticRegression(penalty = LR_Penalty, C = LR_C, max_iter=LR_max_iter)
                itter_times()

def itter_times():
    global v, m, score, maxScore, bestParams, LR_C, LR_Penalty, LR_max_iter, Tfidf_min_df, Tfidf_max_df, Tfidf_max_features
    for i in range(0,1):
        Split_random_state = np.random.randint(2000)#42 , 22

        run_program()

        #print("random_state: ", Split_random_state)
        #print("v, m:", v, m)
        #print("\nScore: ", score[-1])
        #print("\n########################\n")
        if score[-1] > maxScore:
            maxScore = score[-1]
            bestParams = [Tfidf_min_df, Tfidf_max_df, Tfidf_max_features, LR_C, LR_Penalty, LR_max_iter, Split_random_state]     

for p in range(0,10):
    varry_Vec()
    #print("random_state: ", Split_random_state)
    #run_program()         

    Split_random_state = np.random.randint(2000) 
    #print("best params: ", bestParams)
    print("max score", maxScore)
    maxScore = 0
    bestParams = 0



#print("\nMax: ",maxScore)
print("\n--- %s seconds ---" % (time.time() - start_time))