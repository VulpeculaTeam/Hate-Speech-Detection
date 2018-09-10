import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import json
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
import csv
import nltk
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from nltk.corpus import conll2007
import random
#from nltk.corpus import movie_reviews

#da sklearn naive bayes importo di due classificatori Multinomial naive bayse e bernoulli naive bayes
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
#da sklearn linear model importo altri classificatori, la logistiregression, SGDClassifier e BayesianRidge
from sklearn.linear_model import LogisticRegression, SGDClassifier, BayesianRidge
#sempre da sklearn prendo i classificatori Support Vector Machine
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn import datasets, svm, metrics, model_selection, utils
from sklearn import neural_network

from statistics import mode
import pickle
import csv
import re
from itertools import chain
from emoji.unicode_codes import UNICODE_EMOJI
import simplejson
import unidecode
import unicodedata
import treetaggerwrapper 
from pprint import pprint
import itertools
from itertools import groupby
import math
import language_check
from gensim.models import KeyedVectors
import gensim

training = np.genfromtxt('haspeede_FB-testCleanPredict.csv',encoding="utf8", delimiter=',', skip_header=0, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8), dtype=None, invalid_raise = False)

listafrasi2 = [str(x[0]) for x in training]

#lunghezza
lunghezza = [str(x[1]) for x in training]

#percentuale di capslock
capslock = [str(x[2]) for x in training]

#numero di frasi all'interno della frase
numfrasi = [str(x[3]) for x in training]

#numero punti interrogativi esclamativi
numinter = [str(x[4]) for x in training]

#conta le punteggiature all'interno della frase
numpunt = [str(x[5]) for x in training]

#polarita
polarita = [str(x[6]) for x in training]

#percentuale errori
percenterrori = [str(x[7]) for x in training]

#percentuale di parolacce
percentparolacce = [str(x[8]) for x in training]

#percentuale di parolacce
#similarita = [str(x[9]) for x in training]

labels = ['positive', 'negative']

classificazione = []

def convertCommentsToVector(commenti):
    train_x10 = []

    cc = 0
    model = gensim.models.Word2Vec.load("word2vec2final.model")
    for idx,val in enumerate(commenti):
        vettoreMedia = []
        vettoreSomma = []
        for i in range(0,128):
            vettoreMedia.append(0)
            vettoreSomma.append(0)
        cc += 1
        if commenti[idx] != "":
            for r in commenti[idx].split():
                try:
                    vector = model.wv[r] / len(commenti)
                    vettoreMedia = vector + vettoreMedia
                    vettoreSomma = model.wv[r] + vettoreSomma
                except:
                    print("ciao")
        else:
            print(cc)
        
        vettoreMedia = np.append(vettoreMedia, vettoreSomma, axis=0)
        train_x10.append(vettoreMedia)

    return np.array(train_x10)


#listafrasi2 = ["non sembrare vero stare tornare ad essere paese degno chiamare italia"]
testArr = convertCommentsToVector(listafrasi2)

#print(len(testArr))


count = 0
array = []
for t in testArr:

    lst = list(t)

    a = percentparolacce[count]
    a = float(a)
    if math.isnan(a):
        lst.append(0)
    else:
        lst.append(int(round(a*1000)))

    b = polarita[count]
    b = float(b)
    lst.append(int(round(b*100)))
    
    c = numfrasi[count]
    c = float(c)
    lst.append(int(round(c))) 
    
    e = numinter[count]
    e = float(e)
    lst.append(int(round(e)))
    
    f = percenterrori[count]
    f = float(f)
    lst.append(int(round(f)))
    
    g = numpunt[count]
    g = float(g)
    lst.append(int(round(g)))
    
    h = capslock[count]
    h = float(h)
    lst.append(int(round(h*10)))
    
    i = lunghezza[count]
    i = float(i)
    lst.append(int(round(i/10)))
    
    #d = similarita[count]
    #if d == "nan":
    #    d = 0
    #d = float(d)
    #lst.append(int(round(d*100)))
    
    t = np.asarray(lst, dtype=np.float32)
    array.append(t)

    count +=1
    

print("count",count)


array = np.asarray(array)
testArr = array

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# and create a model from that
model = model_from_json(loaded_model_json)
# and weight your nodes with your saved values
model.load_weights('model.h5')

print(testArr)
print("tipo testArr",type(testArr))
count2 = 0
prediction = []
for t in testArr:
    count2 += 1
    lista = []
    lst = list(t)
    lista.append(lst)
    t = np.asarray(lista)
    pred = model.predict(t)
    print(count2,"%s sentiment; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
    if labels[np.argmax(pred)] == "positive":
        classification = 0
        prediction.append(classification)
    else:
        classification = 1
        prediction.append(classification)

commentiOriginali = []

with open("haspeede_FB-test.tsv", encoding="utf8") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
      commentiOriginali.append(row[1])

counter = 0
risultati = []
for c in commentiOriginali:
    c = c + "  ,  " + str(prediction[counter])
    risultati.append(c)
    counter += 1

fh = open("RisultatiFinali.txt", "w", encoding="utf8")

for i in range(0,1000):
    fh.write(str(risultati[i])+"\n")

#etichetteVere = []

#with open("myFile1_1.csv", encoding="utf8") as fd:
#    rd = csv.reader(fd, delimiter=",", quotechar='"')
#    for row in rd:
#      etichetteVere.append(row[2])
#
#print(etichetteVere)
#
#fh = open("prediction.txt", "w") 
#cc3 = 0
#cc4 =0
#for p in prediction:
#    fh.write("id "+str(cc3)+"  "+"etichettaVera: "+str(etichetteVere[cc3])+" "+"etichettaPredetta: "+str(p)+"\n")
#    if (str(etichetteVere[cc3]) != str(p)):
#        cc4 += 1
#        print("errore id: ", cc3)
#    cc3 += 1
#
#print("numero errori", cc4)

#for t in testArr:
#
#    pred = model.predict(t)
#
#    print(np.argmax(pred))
#
#    print("\n")
#    print("%s sentiment; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
#    if labels[np.argmax(pred)] == "positive":
#        classification = 0
#    else:
#        classification = 1
#
#pred = model.predict(testArr, batch_size=None, verbose=1, steps=None)
