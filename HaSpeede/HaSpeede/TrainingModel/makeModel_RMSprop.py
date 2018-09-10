import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import keras.backend as K
import numpy as np
import csv
import simplejson
from tempfile import TemporaryFile
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import time
import os
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

#messages
messages = [str(x[1]) for x in training]

original_messages = []

messagesReader = open("haspeede_FB-train.tsv", encoding="utf8")
rd = csv.reader(messagesReader, delimiter="\t", quotechar='"')

for row in rd:
	original_messages.append(row[0])

train_x10 = []

import gensim
from sklearn import preprocessing, cross_validation, svm

cc = 0
model = gensim.models.Word2Vec.load("word2vec2_128.model")
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
    #featureList.append(a)

    b = polarita[count]
    b = float(b)
    #featureList.append(b)
    lst.append(int(round(b*100)))
    
    c = numfrasi[count]
    c = float(c)
    #featureList.append(c)
    lst.append(int(round(c))) 
    
    e = numinter[count]
    e = float(e)
    #featureList.append(e)
    lst.append(int(round(e)))
    
    f = percenterrori[count]
    f = float(f)
    #featureList.append(f)
    lst.append(int(round(f)))
    
    g = numpunt[count]
    g = float(g)
    #featureList.append(g)
    lst.append(int(round(g)))
    
    h = capslock[count]
    h = float(h)
    #featureList.append(h)
    lst.append(int(round(h*10)))
    
    i = lunghezza[count]
    i = float(i)
    #featureList.append(i)
    lst.append(int(round(i/10)))

    t = np.asarray(lst, dtype=np.float32)
    array.append(t)
    count +=1 

array = np.asarray(array)
train_x = array


X = train_x
Y = train_y


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding

seed = 5

import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(seed)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

np.random.seed(seed)

opt = ["RMSprop"]
rete = [56, 128, 256]
f_a = ["relu","tanh", "sigmoid"]

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
count = 0
fh = open("Risultati.txt", "a")
fcsv = open("Risultati_Csv.txt", "a")  
fcsv.write("Optimizer; Nodi1;  Nodi2;  Funz1;  Funz2;  Acc;  SQM;  Fit;  T_P;  T_N;  F_P;  F_N;  Pre;  Rec;  F1S;  F1S_Macro;" +"\n" +"\n")
cnt = 1
prove_mancanti = 81
for p in range(0, 1):
    for j in range(0,2):
        for k in range(0,1):
            for l in range(0,2):
                for m in range(0,1):
                    cvscores = []
                    over = []
                    y_pred_numpy = []
                    test_index_numpy = []
                    print("******************************************************************")
                    print("Prove Mancanti: ",prove_mancanti)
                    print("******************************************************************")
                    count = 0
                    for train, test in kfold.split(X, Y):
                        count += 1
                        model = Sequential()
                        model.add(Dense(rete[j], input_shape=(256 + numFeatures,), activation=f_a[l]))
                        model.add(Dropout(0.45))
                        model.add(Dense(rete[k], activation=f_a[m]))
                        model.add(Dropout(0.45))
                        model.add(Dense(2, activation='softmax'))

                        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt[p], metrics=['accuracy'])

                        history = model.fit(X[train], Y[train], batch_size = 30,epochs = 25, verbose=1, validation_split=0.03, shuffle=True)

                        y_pred_tmp = model.predict_classes(X[test])
                        test_index_numpy = np.append(test_index_numpy,test)
                        y_pred_numpy = np.append(y_pred_numpy, y_pred_tmp)

                        scores = model.evaluate(X[test],Y[test], batch_size = 30, verbose = 0)

                        print("******************************************************************")
                        print("Prove Mancanti: ",prove_mancanti)
                        print("******************************************************************")

                        print("############################################")
                        print("score: ",count, scores[1])
                        print("############################################")
                        cvscores.append(scores[1] * 100)
                        over.append(scores[1]/history.history["acc"][-1])

                        K.clear_session()

                    prove_mancanti-=1

                    y_pred_list = []
                    for z in range(0,3000):
                        y_pred_list.append(int(y_pred_numpy[z]))


                    test_index_list = []
                    for z in range(0,3000):
                        test_index_list.append(int(test_index_numpy[z]))

                    tmp_list = []
                    for z in range(0,3000):
                        tmp_list.append(str(test_index_list[z])+ str(y_pred_list[z]))

                    results = list(map(int, tmp_list))
                    results.sort()

                    y_pred = []
                    for z in range(0,3000):
                        y_pred.append(abs(results[z]) % 10)

                    y_true = []
                    for z in range(0,3000):
                        y_true.append(train_y[z])

                    tp_list = []
                    tn_list = []
                    fp_list = []
                    fn_list = []

                    tp_list_2 = []
                    tn_list_2 = []
                    fp_list_2 = []
                    fn_list_2 = []

                    for z in range(0,3000):
                        if y_true[z] == 0 and y_pred[z] == 0:
                            tp_list.append(z)
                            tn_list_2.append(z)
                        elif y_true[z] == 1 and y_pred[z] == 1:
                            tn_list.append(z)
                            tp_list_2.append(z)
                        elif y_true[z] == 0 and y_pred[z] == 1:
                            fp_list.append(z)
                            fn_list_2.append(z)
                        elif y_true[z] == 1 and y_pred[z] == 0:
                            fn_list.append(z)
                            fp_list_2.append(z)

                    
                    t_p = len(tp_list)
                    t_n = len(tn_list)
                    f_p = len(fp_list)
                    f_n = len(fn_list)

                    precision = t_p/(t_p + f_p)
                    recall = t_p/(t_p + f_n)
                    f1_score = 2*((precision*recall)/(precision + recall))

                    t_p_2 = len(tp_list_2)
                    t_n_2 = len(tn_list_2)
                    f_p_2 = len(fp_list_2)
                    f_n_2 = len(fn_list_2)

                    precision_2 = t_p_2/(t_p_2 + f_p_2)
                    recall_2 = t_p_2/(t_p_2 + f_n_2)
                    f1_score_2 = 2*((precision_2*recall_2)/(precision_2 + recall_2))

                    f1_score_macro = (f1_score + f1_score_2)/2

                    directory = 'C:\\Users\\Giuli\\Desktop\\HaSpeede\\HaSpeede\\TrainingModel\\Risultati\\' + str(opt[p]) + "_" + str(rete[j]) + "_" + str(rete[k]) + "_" + str(f_a[l]) + "_" + str(f_a[m])
                    
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    tp_file = open(directory + "\\truePositive.txt", "a", encoding="utf8")
                    for i in range(0,len(tp_list)):
                        tp_file.write(str(int(tp_list[i]+1)) + "  ;  " + messages[int(tp_list[i])] + "  ;  " + original_messages[int(tp_list[i])] + "\n")
                    tp_file.close()

                    tn_file = open(directory + "\\trueNegative.txt", "a", encoding="utf8")
                    for i in range(0,len(tn_list)):
                        tn_file.write(str(int(tn_list[i]+1)) + "  ;  " + messages[int(tn_list[i])] + "  ;  " + original_messages[int(tn_list[i])] + "\n")
                    tn_file.close()

                    fp_file = open(directory + "\\falsePositive.txt", "a", encoding="utf8")
                    for i in range(0,len(fp_list)):
                        fp_file.write(str(int(fp_list[i]+1)) + "  ;  " + messages[int(fp_list[i])] + "  ;  " + original_messages[int(fp_list[i])] + "\n")
                    fp_file.close()

                    fn_file = open(directory + "\\falseNegative.txt", "a", encoding="utf8")
                    for i in range(0,len(fn_list)):
                        fn_file.write(str(int(fn_list[i]+1)) + "  ;  " + messages[int(fn_list[i])] + "  ;  " + original_messages[int(fn_list[i])] + "\n")
                    fn_file.close()

                    fcsv.write(str(opt[p])+";"+str(rete[j])+";"+str(rete[k])+";"+str(f_a[l])+";"+str(f_a[m])+";"+str("%.2f" % np.mean(cvscores))+";"+str("%.2f" % np.std(cvscores))+";"+str("%.2f" % np.mean(over))+";"+str("%.0f" % t_p)+";"+str("%.0f" % t_n)+";"+str("%.0f" % f_p)+";"+str("%.0f" % f_n)+";"+str("%.3f" % precision)+";"+str("%.3f" % recall)+";"+str("%.3f" % f1_score)+";"+str("%.3f" % f1_score_macro)+"\n")
                    fh.write("Optimizer: "+str(opt[p])+" Nodi1: "+str(rete[j])+" Nodi2: "+str(rete[k])+" Funz1: "+str(f_a[l])+" Funz2: "+str(f_a[m])+" Acc: "+str("%.2f" % np.mean(cvscores))+" SQM: "+str("%.2f" % np.std(cvscores))+" Fit: "+str("%.2f" % np.mean(over))+" T_P: "+str("%.0f" % t_p)+" T_N:"+str("%.0f" % t_n)+" F_P: "+str("%.0f" % f_p)+" F_N: "+str("%.0f" % f_n)+" Pre: "+str("%.3f" % precision)+" Rec: "+str("%.3f" % recall)+" F1S:"+str("%.3f" % f1_score)+" F1S_Macro:"+str("%.3f" % f1_score)+"\n")
                    
                    K.clear_session()

'''

print("############################################")
print(y_pred)
print(len(y_pred))
y_pred2 = model.predict_classes(X)
print("############################################")
print(y_pred2)
print(len(y_pred2))
print("############################################")
print(np.array_equal(y_pred,y_pred2))
print("############################################")

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')
print('saved model!')
fh.close() 
'''
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


