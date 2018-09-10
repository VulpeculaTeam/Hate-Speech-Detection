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
import we as we

tool = language_check.LanguageTool('it-IT')

def deleteDuplicate (riga):
    parole = riga.split()
    stringa = ""
    for a in parole:
        a = [list(g) for k, g in groupby(a)]    

        vocali = ['a','e','i','o','u','y']
        
        for idx,val in enumerate(a):
            if idx == 0:
                stringa += a[idx][0] 
            elif idx == len(a)-1:
                stringa += a[idx][0]
            elif a[idx][0] in vocali:
                stringa += a[idx][0]
            elif len(a[idx]) == 1:
                stringa += a[idx][0]
            elif len (a[idx]) >= 2:
                stringa += a[idx][0]
                stringa += a[idx][1]
        stringa =  stringa + " "
        
    return(stringa)  

#funzione che rimuove gli accenti dal testo
def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def text_to_id(text):
    text = strip_accents(text.lower())
    return text

#le stopWords italiane
nltk.download('stopwords')
nltk.download('punkt')
stopWords = set(nltk.corpus.stopwords.words('italian'))

train = []
emoji_pattern = re.compile("["
       # u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

congiunzioni = [' e ', ' né ', ' n ', ' ne ', ' o ', ' inoltre ', ' ma ', ' però ', ' pero ', ' dunque ', ' anzi ', ' che ', ' allorché ',  ' allorchè ',  ' allorche ', ' perché ',' perche ',' perchè ', ' giacché ', ' giacchè ', ' purché ', ' purchè ',' purche ',' affinché ',  ' affinchè ',  ' affinche ', ' eppure ', ' oppure ', ' dopoché ',  ' dopochè ', ' dopoche ', ' con tutto ciò ',  ' con tutto cio ', ' di modo che ', ' in quanto che ', ' anche ', ' neanche ', ' neppure ' , ' ovvero ', ' oppure ', ' tuttavia ', ' anzi ',' infatti ', ' cioè ', ' cioé ', ' cioe ', ' ossia ', ' dunque ', ' quindi ', ' perciò ', ' percio ']

preposizioni = [' di ', ' a ', ' da ', ' in ', ' con ', ' su ', ' per ', ' tra ', ' fra ', ' al ', ' allo ', ' all ', ' alla ', ' ai ', ' agli ', ' le ', ' dal ', ' dallo ' ,' dall ',' dalla ', ' dai ', ' dagli ', ' dalle ', ' de ', ' del ', ' dello ', ' dell ', ' della ' ,' dei ', ' degli ', ' delle ', ' ne ', ' nel ', ' nello ', ' nell ', ' nella ', ' nei ',' negli ', ' nelle ', ' sul ',' sullo ',  ' sull ', ' sulla ', ' sui ' ,' sugli ', ' sulle ']

articoli = [' il ',' l ',' lo ',' i ',' gli ',' la ',' le ',' un ',' uno ',' una ']

a = [' 1 ',' 6 ',' bll ',' bn ',' c ',' cm ',' cmq ',' cmnq ',' cn ',' cpt ',' grz ',' k ',' ke ',' msg ', ' nn ',' nnt ',' qlcs ',' qnd ',' sn ',' tt ',' v ',' x ',' xche ',' xchè ',' xché ',' xk ',' xke ',' xké ',' xkè ',' xo ', ' xò ', ' xó ', ' qlc ', ' qlcn ',' ste ', ' sta ', ' sto ', ' sti ', ' col ', ' + ', ' - ', ' = ']

b = [' uno ', ' sei ',' bello ',' bene ',' ci ',' come ',' comunque ',' comunque ',' con ',' capito ',' grazie ',' che ',' che ',' messaggio ',' non ',' niente ',' qualcosa ',' quando ',' sono ',' tutto ',' vi ',' per ',' perché ',' perché ',' perché ',' perché ',' perché ',' perché ',' perché ',' però ', ' però ',' però ', ' qualche ', ' qualcuno ', ' queste ', ' questa ', ' questo ', ' questi ', ' con il ', ' più ', ' meno', ' uguale ' ]

faccine = [':p', ':)', ':(' ,':/' ,'xd' ,':-(' ,':-)', ':-d', '-_-', ':-*'];

risate = ['ah', 'ha', 'eh', 'he' 'ih', 'hi']

numeriLettere = [' uno ', ' due ', ' tre ' ' quattro ' , ' cinque ', ' sette ', ' otto ', ' nove ', ' dieci ', ' venti ', ' trenta ', ' quaranta ', ' cinquanta ', ' sessanta ', ' settanta ', ' ottanta ', ' novanta ', ' cento ', ' mille ']

pronomi = [' io ', ' tu ', ' egli ', ' noi ', ' voi ', ' essi' , ' me ',' te ',' lui ',' se ',' cio ',' noi ',' voi ',' essi ',' loro ',' li ',' vi ',' ci ',' mi ',' ti ',' la ',' le ',' ne ',' gli ',' lo ',' lei ',' si ']

#commentiok = []
#import time
#with open("myFile1_1.csv") as fd:
#    rd = csv.reader(fd, delimiter=",", quotechar='"')
#
#    for row in rd:
#    	commentiok.append(row[0])
#
#print(commentiok)
#
#with open("commentipuliti.csv","w",newline="",encoding="utf8") as f:  
#    cw = csv.writer(f)
#    cw.writerows([r] for r in commentiok)
#
#print("finito")
#time.sleep(20)
'''
parole = []

spamReader = csv.reader(open('parole_plus_parolacce.csv', newline=''), delimiter=' ', quotechar='|')

for row in spamReader:
	parole.append(row)

count = 0
with open("200kcommentiunclean.csv", encoding="utf8") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')

    for row in rd:
    	if row != []:
	        #tutte le lettere diventano minuscole
	        row[0] = row[0].lower()
	        #scambiare la chioccola con la a
	        row[0] = row[0].replace('@', 'a')

	        #scambiare la & con la e
	        row[0] = row[0].replace('&', 'e')

	        #trova le parolacce camuffate
	        row[0] = we.convertiParolacce(row[0])

	        #se nel commento è presento un hashtag, richiama la funzione di split che splitta l'hashtag
	        if "#" in row[0]:
	            row[0] = we.richiamaSplit(row[0])

	        #aggiungere uno spazio all'inzio e alla fine della frase
	        row[0] = " " + row[0] + " "

	        #rimuovere i link
	        row[0] = re.sub(r'http\S+', ' ', row[0])

	        #elimina doppie
	        row[0] = deleteDuplicate(row[0]) 
	        row[0] = " " + row[0] + " "

	        #rimuovere punteggiatura
	        pat2 = re.compile(r"([.()!?|,:-;])")
	        row[0] = pat2.sub(' ', row[0])

	        stringa = we.convertiFaccine(row[0])

	        #rimuovere i symbols & pictographs, transport & map symbols e flags (iOS)
	        row[0] = emoji_pattern.sub(r'', row[0])

	        #inserire gli spazi prima e dopo una emoji
	        pat = re.compile('([\U0001F602-\U0001F64F])')
	        row[0] = pat.sub(" \\1 " , row[0])    
	        row[0] = pat.sub(" " , row[0])  
	        
	        #rimuovere le emoji
	        row[0] = unidecode.unidecode(row[0])
	        
	        #rimuovere gli hashtag
	        row[0] = row[0].replace('#', ' ')

	        #sostituire le abbreviazioni alle rispettive parole
	        for idx,val in enumerate(a):
	            row[0] = row[0].replace(val, b[idx])
	        
	        #rimuovere le congiunzioni
	        for word in congiunzioni:
	            row[0] = row[0].replace(word, ' ')

	        #rimuovere le preposizioni
	        for word in preposizioni:
	            row[0] = row[0].replace(word, ' ')

	        #rimuovere gli articoli
	        for word in articoli:
	            row[0] = row[0].replace(word, ' ')

	    	#rimuovere i pronomi
	        for word in pronomi:
	            row[0] = row[0].replace(word, ' ')

	        #rimuovere le faccine composte da caratteri (non emoticons)
	        for word in faccine:
	            row[0] = row[0].replace(word, ' ')

	        #replace numeri in "NUM"
	        row[0] = re.sub("\d+", " ", row[0])

	        #rimuovere i numeri scritti in lettere
	        for word in numeriLettere:
	            row[0] = row[0].replace(word, ' ')
	        
	        #rimuovere gli hashtag
	        row[0] = row[0].replace('\"', ' ')
	        
	        #rimuovere le risate
	        row[0] = we.rimuoviRisate(row[0])

	        if len(row[0]) != 1:
	            #lemmatizziamo la frase
	            row[0] = we.lemmatizza(row[0])

	        #row[0] = we.similaritaParola(row[0])
	        #rimuovere gli accenti nuovamente
	        row[0] = text_to_id(row[0])

	        #rimuovere i trattini
	        row[0] = row[0].replace('-', ' ')

	        #sostituisce gli errori o le parole sconosciute con le parole più simili che trova nel vocabolario
        	row[0] = we.similaritaParola(row[0], parole)

	        #print(row)
	        #aggiungere la riga alla lista
	        row[0] = row[0]+stringa
	        print(count)
	        count +=1
	        if count == 1000:
	        	print("1000")
	        if count == 20000:
	        	print("20000")
	        if count == 40000:
	        	print("40000")
	        if count == 60000:
	        	print("60000")
	        if count == 80000:
	        	print("80000")
	        if count == 100000:
	        	print("100000")
	        if count == 130000:
	        	print("130000")
	        if count == 160000:
	        	print("160000")
	        if count == 180000:
	        	print("180000")

	        train.append(row)
	     

#salvo la lista pulita in un file csv
with open("200kcommenticlean.csv","w",newline="",encoding="utf8") as f:  
    cw = csv.writer(f)
    cw.writerows(r for r in train)
'''
import gzip
import gensim 
import logging
import csv
import time
import numpy as np
import smart_open

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
parole4 = []
with open("200kcommenticlean.csv",encoding="utf8") as fd:
	rd = csv.reader(fd, delimiter=",", quotechar='"')

	for row in rd:
		riga = []
		if row != []:
			riga.append(row[0].split())
			parole4.append(riga)

	#for row in rd:
	#	parole4.append(row)

#def read_corpus(fname, tokens_only=False):
#    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
#        for i, line in enumerate(f):
#            if tokens_only:
#                yield gensim.utils.simple_preprocess(line)
#            else:
#                # For training data, add tags
#                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])
#
#train_corpus = list(read_corpus("allcomment.csv"))
#print(train_corpus)
#print(parole4)


flat_list = []
for sublist in parole4:
	for item in sublist:
		flat_list.append(item) 


#min_count è lo soglia cioè se la parola è stata trovata meno di 2 volte viene tolta dal dizionario 
#min_count per default è 5
model = gensim.models.Word2Vec(flat_list, size=128, window=20, min_count=1, workers=4)
model.train(flat_list, total_examples=len((flat_list)), epochs=25)
#model = gensim.models.Doc2Vec(train_corpus, size=128, window=20, min_count=1, workers=4)
#print("\n###############\nora ci alleniamo\n##############\n")
#model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
##
model.save("word2vec2final.model")
#model = gensim.models.Word2Vec.load('it.bin')
#model = gensim.models.Word2Vec.load("word2vec.model")
#vector = model.wv['coglionedsj']
print("vettore della parola coglione \n")
vector = model.wv['coglione']
print(vector)
np.set_printoptions(threshold=np.nan)
#print(len(vector))
#print(vector)
print('Found %s word vectors of word2vec' % len(model.wv.vocab))
try:
	print(model.most_similar(positive="coglione"))
except:
	print("parola non trovata")
