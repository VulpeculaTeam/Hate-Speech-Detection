
import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import numpy as np
import csv
import simplejson
from tempfile import TemporaryFile
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import time
from gensim.models import KeyedVectors

training = np.genfromtxt('myFile1_1.csv',encoding="utf8", delimiter=',', skip_header=0, usecols=(2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11), dtype=None, invalid_raise = False)

#lunghezza
lunghezza = [str(x[2]) for x in training]

#percentuale di capslock
capslock = [str(x[3]) for x in training]

#numero di frasi all'interno della frase
numfrasi = [str(x[4]) for x in training]

#numero punti interrogativi esclamativi
numinter = [str(x[5]) for x in training]

#conta le punteggiature all'interno della frase
numpunt = [str(x[6]) for x in training]

#polarita
polarita = [str(x[7]) for x in training]

#percentuale errori
percenterrori = [str(x[8]) for x in training]

#percentuale di parolacce
percentparolacce = [str(x[9]) for x in training]

#similarita
similarita = [str(x[10]) for x in training]
# creare il nostro training data dai tutti i commenti
train_x = [str(x[1]) for x in training]

print(type(train_x))


train_x10 = []

import gensim
from sklearn import preprocessing, cross_validation, svm

cc = 0
model = gensim.models.Word2Vec.load("word2vec2_128.model")
modelvec = gensim.models.Word2Vec.load("word2vec2_128.model")
for idx,val in enumerate(train_x):
    vettoreMedia = []
    vettoreSomma = []
    for i in range(0,128):
        vettoreMedia.append(0)
        vettoreSomma.append(0)

    if train_x[idx] != "":
        for r in train_x[idx].split():           
            vector = model.wv[r] / len(train_x)
            vettoreMedia = vector + vettoreMedia
            vettoreSomma = model.wv[r] + vettoreSomma            
    else:
        cc += 1

    vettoreMedia = np.append(vettoreMedia, vettoreSomma, axis=0)
    train_x10.append(vettoreMedia)

train_x = np.array(train_x10)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

#train_x = preprocessing.scale(train_x)
#train_x = preprocessing.normalize(train_x, norm='l2')
#train_x = min_max_scaler.fit_transform(train_x)

# indicizza tutte le etichette dei commenti (o 0 1)
train_y = np.asarray([int(x[0]) for x in training])

np.set_printoptions(threshold=np.nan)
   
numFeatures = 8
count = 0
array = []
for t in train_x:
    lst = list(t)
    featureList = []

    a = percentparolacce[count]
    a = float(a)
    lst.append(int(round(a*1000)))
    featureList.append(int(round(a*1000)))

    b = polarita[count]
    b = float(b)
    featureList.append(int(round(b*100)))
    lst.append(int(round(b*100)))
    
    c = numfrasi[count]
    c = float(c)
    featureList.append(int(round(c)))
    lst.append(int(round(c))) 
    
    e = numinter[count]
    e = float(e)
    featureList.append(int(round(e)))
    lst.append(int(round(e)))
    
    f = percenterrori[count]
    f = float(f)
    featureList.append(int(round(f)))
    lst.append(int(round(f)))
    
    g = numpunt[count]
    g = float(g)
    featureList.append(int(round(g)))
    lst.append(int(round(g)))
    
    h = capslock[count]
    h = float(h)
    featureList.append(int(round(h*10)))
    lst.append(int(round(h*10)))
    
    i = lunghezza[count]
    i = float(i)
    featureList.append(int(round(i/10)))
    lst.append(int(round(i/10)))

    #normalize = [float(i)/sum(featureList) for i in featureList]

     #d = similarita[count]
     #if d == "nan":
     #    d = 0
     #d = float(d)
    #lst.append(int(round(d*100)))
    #numpfeature = np.asarray(featureList, dtype=np.float32)
    #numpfeature = preprocessing.scale(numpfeature)
    #lstFeature = list(numpfeature)
    #print(len(lst))
    #print(len(lstFeature))
    #lst = lst+lstFeature
    print(count, featureList)
    t = np.asarray(lst, dtype=np.float32)
    array.append(t)
    count +=1 

array = np.asarray(array)
train_x = array

#scaler = preprocessing.StandardScaler().fit(train_x)
#train_x = min_max_scaler.fit_transform(train_x)

#train_x = preprocessing.scale(train_x)


X = train_x
Y = train_y


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding

seed = 7
np.random.seed(seed)

f_a = ["Adagrad"]
rete = [56, 128, 256]
f_b = ["relu","tanh", "sigmoid"]

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
count = 0
fh = open("tmp.txt", "a") 
cnt = 1
                
cvscores = []
over = []
for i in range(0, 1):
    cvscores = []
    over = []
    for train, test in kfold.split(X, Y):
        count += 1
        model = Sequential()
        #print(str(rete[i])+";"+str(rete[j])+";"+str(f_a[k])+";"+str(f_a[m])+";"+str(np.mean(cvscores))+";"+str(np.std(cvscores))+";"+str(np.mean(over))+"\n")
        model.add(Dense(256, input_shape=(256 + numFeatures,), activation="relu"))
        model.add(Dropout(0.45))
        model.add(Dense(56, activation="sigmoid"))
        model.add(Dropout(0.45))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer="Adagrad", metrics=['accuracy'])

        #print("abbbbbaa",len(X[test]))
        history = model.fit(X[train], Y[train], batch_size = 25,epochs = 35, verbose=1, validation_split=0.03, shuffle=True)

        scores = model.evaluate(X[test],Y[test], batch_size = 25, verbose = 0)

        #ynew = model.predict_classes(X[test])
        ## show the inputs and predicted outputs
        ##print("X=%s, Predicted=%s" % (X[test][0], ynew[0]))
        #print("X=%s, Predicted=%s" % (X[test][2], ynew[2]))
        #print(type(X[test]))
        #vec = X[test][2].tolist()
        #print(vec[-8:])
        #vec = vec[:-8]
        #import time
        #time.sleep(100)

        #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        print("############################################")
        #print("optimizer:",f_a[k],"numero nodi primo livello",rete[i],"numero nodi secondo livello",rete[j],"\n")
        print("score: ",count, scores[1])
        print("############################################")
        cvscores.append(scores[1] * 100)
        over.append(scores[1]/history.history["acc"][-1])

        #fg.write(str("%.2f" % scores[1])+"\n")
        
    #Nodi_primo_hidden_layer   #Nodi_secondo_hidden_layer   #Funzione_att_primo_hlayer  #Funzione_att_secondo_hlayer  #media_acc  #scarco_quad_medio   #livello_overf
    #fh.write(str(rete[i])+";"+str(rete[j])+";"+str(f_b[k])+";"+str(f_b[l])+";"+str("%.2f" % np.mean(cvscores))+";"+str("%.2f" % np.std(cvscores))+";"+str("%.2f" % np.mean(over))+"\n")

    #fh.write(str(rete[i])+";"+str(rete[j])+";"+str(f_b[k])+";"+str(f_b[l]))
    fh.write(str("%.2f" % np.mean(cvscores))+";"+str("%.2f" % np.std(cvscores))+";"+str("%.2f" % np.mean(over))+"\n")


model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')
print('saved model!')
fh.close() 

'''


import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import numpy as np
import csv
import simplejson
from tempfile import TemporaryFile
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import time
from gensim.models import KeyedVectors

training = np.genfromtxt('lista.csv',encoding="utf8", delimiter=';', skip_header=0, usecols=(2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), dtype=None, invalid_raise = False)

#lunghezza
lunghezza = [str(x[2]) for x in training]

#percentuale di capslock
capslock = [str(x[3]) for x in training]

#numero di frasi all'interno della frase
numfrasi = [str(x[4]) for x in training]

#numero punti interrogativi esclamativi
numinter = [str(x[5]) for x in training]

#conta le punteggiature all'interno della frase
numpunt = [str(x[6]) for x in training]

#polarita
polarita = [str(x[7]) for x in training]

#percentuale errori
percenterrori = [str(x[8]) for x in training]

#percentuale di parolacce
percentparolacce = [str(x[9]) for x in training]

#similarita
similarita = [str(x[10]) for x in training]

# creare il nostro training data dai tutti i commenti
train_x = [str(x[1]) for x in training]

train_x10 = []

import gensim
cc = 0
model = gensim.models.Word2Vec.load("word2vec2.model")
for idx,val in enumerate(train_x):
    vettoreMedia = []
    vettoreSomma = []
    for i in range(0,128):
        vettoreMedia.append(0)
        vettoreSomma.append(0)

    if train_x[idx] != "":
        for r in train_x[idx].split():
            try:
                vector = model.wv[r] / len(train_x)
                vettoreMedia = vector + vettoreMedia
                vettoreSomma = model.wv[r] + vettoreSomma
            except:
                print("ciao")
    else:
        cc += 1

    vettoreMedia = np.append(vettoreMedia, vettoreSomma, axis=0)
    train_x10.append(vettoreMedia)

train_x = np.array(train_x10)
# indicizza tutte le etichette dei commenti (o 0 1)
train_y = np.asarray([int(x[0]) for x in training])

np.set_printoptions(threshold=np.nan)

print(train_x[:3])
print(len(train_x[1]))

        
numFeatures = 9
count = 0
array = []
for t in train_x:
    lst = list(t)

    a = percentparolacce[count]
    a = float(a)
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
    
    d = similarita[count]
    if d == "nan":
        d = 0
    d = float(d)
    lst.append(int(round(d*100)))
    

    t = np.asarray(lst, dtype=np.float32)
    array.append(t)
    count +=1 

array = np.asarray(array)
train_x = array

X = train_x
Y = train_y

from sklearn import preprocessing, cross_validation, svm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding

seed = 7
np.random.seed(seed)

f_a = ["Adagrad"]
rete = [56, 128, 256]
f_b = ["relu","tanh", "sigmoid"]

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
count = 0
fh = open("risultati_sigmoid.txt", "a") 
cnt = 1

cvscores = []
over = []
for train, test in kfold.split(X, Y):
    count += 1
    model = Sequential()
    #print(str(rete[i])+";"+str(rete[j])+";"+str(f_a[k])+";"+str(f_a[m])+";"+str(np.mean(cvscores))+";"+str(np.std(cvscores))+";"+str(np.mean(over))+"\n")
    model.add(Dense(128, input_shape=(256 + numFeatures,), activation="sigmoid"))
    model.add(Dropout(0.45))
    model.add(Dense(56, activation="sigmoid"))
    model.add(Dropout(0.45))
    #model.add(Dense(56, activation="sigmoid"))
    #model.add(Dropout(0.45))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer="Adagrad", metrics=['accuracy'])

    #print("abbbbbaa",len(X[test]))
    history = model.fit(X[train], Y[train], batch_size = 30,epochs = 25, verbose=1, validation_split=0.03, shuffle=True)

    scores = model.evaluate(X[test],Y[test], batch_size = 30, verbose = 0)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("############################################")
    #print("optimizer:",f_a[k],"numero nodi primo livello",rete[i],"numero nodi secondo livello",rete[j],"\n")
    print("score: ",count, scores[1])
    print("############################################")
    cvscores.append(scores[1] * 100)
    over.append(scores[1]/history.history["acc"][-1])
    #fg.write(str("%.2f" % scores[1])+"\n")
#Nodi_primo_hidden_layer   #Nodi_secondo_hidden_layer   #Funzione_att_primo_hlayer  #Funzione_att_secondo_hlayer  #media_acc  #scarco_quad_medio   #livello_overf
#fh.write(str(rete[i])+";"+str(rete[j])+";"+str(f_b[k])+";"+str(f_b[l])+";"+str("%.2f" % np.mean(cvscores))+";"+str("%.2f" % np.std(cvscores))+";"+str("%.2f" % np.mean(over))+"\n")
fh.write(str("%.2f" % np.mean(cvscores))+";"+str("%.2f" % np.std(cvscores))+";"+str("%.2f" % np.mean(over))+"\n")

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')
print('saved model!')
fh.close()
''' 


