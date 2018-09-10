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

'''
#perchè questa classe?
#dato che vi sono svariati algoritmi per la classificazione, perchè utilizzarne solo uno?
#Si possono combinare algoritmi per la classificazione, cercando di creare una sorta di sistema
#di votazione in cui ogni algoritmo da un voto e la classificazione che ha la maggioranza dei voti
#è quella che viene scelta

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        #per ogni classificatore, chiediamo di classificare la frase basandoci sulle features
        #terminato il ciclo for, la return restituisce il voto più popolare
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    #la funzione classify potrebbe anche bastare ai fini della classificazione, ma utilizzando la funzione
    #sottostante otteniamo un altro parametro che è la confidenza
    #dato che ogni classificatore esprime il suo parere riguardo la frase da classificatore (positiva o negativa
    #1 o 0) si possono calcolare i voti a favore e contro.
    #Supponiamo di usare 5 classificatori, il nostro commento verrà classificato 5 volte e avrà 5 voti.
    #Se per tutti i classificatori la frase è negativa, si avranno 5 voti su 5 e il livello di confidenza sarà 1.0 (5/5)
    #Se invece ottiene 3 voti negativi (1) e 3 voti positivi (0) la confidenza sarà (3/5)
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


'''
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

k = 1

spamReader = csv.reader(open('parole_plus_parolacce.csv', newline=''), delimiter=' ', quotechar='|')

parole = []

for row in spamReader:
    parole.append(row)

counter = 0;

with open("haspeede_TW-test.tsv", encoding="utf8") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')

    for row in rd:

        #1
        #-----------------------------------------------------#
        #conta la lunghezza della frase e lo aggiunge come feature
        row = row + we.lunghezzaFrase(row[k])
        #-----------------------------------------------------#

        #2
        #-----------------------------------------------------#
        #conta il numero di capslock e lo aggiunge come feature
        row = row + we.cercaCapsLock(row[k])
        #-----------------------------------------------------#

        #tutte le lettere diventano minuscole
        row[k] = row[k].lower()

        #per twitter
        row[k] = we.richiamaSplitAt(row[k])

        #scambiare la chioccola con la a
        row[k] = row[k].replace('@', 'a')

        #scambiare la & con la e
        row[k] = row[k].replace('&', 'e')

        #trova le parolacce camuffate
        row[k] = we.convertiParolacce(row[k])

        #se nel commento è presento un hashtag, richiama la funzione di split che splitta l'hashtag
        if "#" in row[k]:
            row[k] = we.richiamaSplit(row[k])

        #aggiungere uno spazio all'inzio e alla fine della frase
        row[k] = " " + row[k] + " "

        #rimuovere i link
        row[k] = re.sub(r'http\S+', 'link', row[k])

        #elimina doppie
        row[k] = deleteDuplicate(row[k]) 
        row[k] = " " + row[k] + " "

        #3
        #-----------------------------------------------------#
        #conta il numero di frasi all'interno del commento fratto il numero di parole totali
        row = row + we.countSentences(row[k])
        #-----------------------------------------------------#

        #4
        #-----------------------------------------------------#
        #conta il numero di punti interrogativi e esclamativi
        row = row + we.numeroInterrEscl(row[k])
        #-----------------------------------------------------#

        #5
        #-----------------------------------------------------#
        #conta le punteggiature (. ,) all'interno della frase
        row = row + we.numeroPunteggiatura(row[k])
        #-----------------------------------------------------#

        #rimuovere punteggiatura
        pat2 = re.compile(r"([.()!?|,:-;`''\\//<>_^|%$£«»*“=´+])")
        row[k] = pat2.sub(' ', row[k])

        stringa = we.convertiFaccine(row[k])

        #rimuovere i symbols & pictographs, transport & map symbols e flags (iOS)
        row[k] = emoji_pattern.sub(r'', row[k])

        #inserire gli spazi prima e dopo una emoji
        pat = re.compile('([\U0001F602-\U0001F64F])')
        row[k] = pat.sub(" \\1 " , row[k])    
        row[k] = pat.sub(" " , row[k])  

        #rimuovere le emoji
        row[k] = unidecode.unidecode(row[k])
        
        #rimuovere gli hashtag
        row[k] = row[k].replace('#', ' ')

        #sostituire le abbreviazioni alle rispettive parole
        for idx,val in enumerate(a):
            row[k] = row[k].replace(val, b[idx])
        
        #rimuovere le congiunzioni
        for word in congiunzioni:
            row[k] = row[k].replace(word, ' ')

        #rimuovere le preposizioni
        for word in preposizioni:
            row[k] = row[k].replace(word, ' ')

        #rimuovere gli articoli
        for word in articoli:
            row[k] = row[k].replace(word, ' ')

    	#rimuovere i pronomi
        for word in pronomi:
            row[k] = row[k].replace(word, ' ')

        #rimuovere le faccine composte da caratteri (non emoticons)
        for word in faccine:
            row[k] = row[k].replace(word, ' ')

        #replace numeri in "NUM"
        row[k] = re.sub("\d+", " ", row[k])

        #rimuovere i numeri scritti in lettere
        for word in numeriLettere:
            row[k] = row[k].replace(word, ' ')
        
        #rimuovere gli hashtag
        row[k] = row[k].replace('\"', ' ')
        
        #rimuovere le risate
        row[k] = we.rimuoviRisate(row[k])

        if len(row[k]) != 0:
            #lemmatizziamo la frase
            row[k] = we.lemmatizza(row[k])

        #6
        #-----------------------------------------------------#
        #calcolare la polarità della frase come feature
        row = row + we.polaritaFrase(row[k])
        #-----------------------------------------------------#
        
        #-----------------------------------------------------#

        #rimuovere gli accenti nuovamente
        row[k] = text_to_id(row[k])

        #rimuovere i trattini
        row[k] = row[k].replace('-', ' ')

        #7
        if len(row[k]) != 0:
        ##aggiunge la percentuale di errori all'interno della frase come feature
            row = row + we.percentualeErrori(row[k])
        #-----------------------------------------------------#

        pat2 = re.compile(r"([.()!?|,:-;`''\\//<>_^|%$£«»*“=´❤+])")
        row[k] = pat2.sub(' ', row[k])

        print(row[k])
        #sostituisce gli errori o le parole sconosciute con le parole più simili che trova nel vocabolario
        row[k] = we.similaritaParola(row[k], parole)

        row[k] = pat2.sub(' ', row[k])
        #8
        #-----------------------------------------------------#
        #aggiunge alla riga il numero di parolacce percentuale all'interno di una frase come feature
        if len(row[k].split()) != 0:
            row = row + we.trovaParolacce(row[k])
        else:
            row = row + [str(0)]
        
        #9
        #aggiunge il numero di parolacce all'interno della frase
        if len(row[k].split()) != 0:
            row = row + we.contaParolacce(row[k])
        else:
            row = row + [str(0)]
        #-----------------------------------------------------#

        #9
        #-----------------------------------------------------#
        #trova la similarità
        #row = row + we.calcolaSimilarita(row[k])
        #-----------------------------------------------------#

        #10
        #-----------------------------------------------------#
        #calcolare la polarità della frase secondo textBlob
        row = row + we.polaritaTextBlob(row[k])
        #-----------------------------------------------------#

        #11
        #-----------------------------------------------------#
        #calcolare la subjectivity della frase secondo textBlob
        row = row + we.soggettivitaTextBlob(row[k])
        #-----------------------------------------------------#

        #print(row)
        #aggiungere la riga alla lista
        row[k] = row[k]+stringa

        print(counter,row)
        counter += 1
        train.append(row)

#salvo la lista pulita in un file csv
with open("haspeede_TW-testClean.csv","w",newline="",encoding="utf8") as f:  
    cw = csv.writer(f)
    cw.writerows(r+[""] for r in train)
'''
#salvare in un file txt le frase pulite
#si otterrà una lista di liste formate da
#["id", "commento_pulito", "0/1"]
f = open('output.txt', 'w')
simplejson.dump(train, f)
f.close()

#togliere l'id
for sublist in train:
    del sublist[0]


#le seguenti operazioni le faccio per ottenere
#una lista di liste in cui il primo elemento 
#è una lista formata da una lista contente le parole del
#commento separate, e la classificazione
giudizio = []
for y in train:
    giudizio.append(y[1])
frasi = []

#split frase
for x in train:
    frase = x[0].split()
    frasi.append(frase)

d = open('frasi.txt', 'w')
simplejson.dump(frasi, d)
d.close()

finale = []
for i in range(len(train)):
    finale.append(tuple([frasi[i], giudizio[i]]))


#salvare in un file txt la lista di liste
#appena ottenuta
d = open('finale.txt', 'w')
simplejson.dump(finale, d)
d.close()

documents = finale

#tagger permette di associare un tag ad ogni parola
tagger = treetaggerwrapper.TreeTagger(TAGLANG="it")

parole1 = []
parole = []
for sublist in frasi:
    for item in sublist:
        #tutti minuscolo, anche se non c'è bisogno
        item.lower()
        #funzione tokenize che è uno scanner lessicale
        item = word_tokenize(item)
        #taggo ogni parola
        pos1 = tagger.tag_text(item)
        #creo tuple formate da commento, tag e lemma
        pos = treetaggerwrapper.make_tags(pos1)
        
        for w in pos:
            #se il tag è aggettivo, nome o verbo, lo voglio nell'insieme di features
            if w[1] == "ADJ" or w[1] == "NOM" or w[1] == "VER":
                #aggiungo il lemma della parola e non la parola
                parole1.append(w[2].lower())
        #parole.append(item[0])

it = nltk.stem.snowball.ItalianStemmer()
for w in parole1:
    #a questo punto controllo se la parola fa parte delle stopWords 
    #e se non ne fa parte la aggiungo
    if w not in stopWords:
        #w = it.stem(w)
        parole.append(w)


#salvare le parole in un file txt
salva_parole = open("parole.pickle","wb")
pickle.dump(parole1, salva_parole)
salva_parole.close()

#convertire una lista di parole in un nltk frequency distribution
parole = nltk.FreqDist(parole)
#stampare le 15 parole più utilizzate
print(parole.most_common(15))
#prendere le 3000 parole più popolari
word_features = list(parole.keys())

d = open('parole.txt', 'w')
simplejson.dump(parole1, d)
d.close()

d = open('wordfeatures.txt', 'w')
simplejson.dump(word_features, d)
d.close()

print(len(word_features))

save_documents = open("documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

save_word_features = open("word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

#funzione che trova le features nel documents, cioè la prima lista di liste che
#è il risultato della prima pulizia del dataset
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

save_featuresets = open("featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

#tramite la random.shuffle facciamo si che il train e il test siano sempre diversi 
random.shuffle(featuresets)

t = open('featuresets.txt', 'w')
simplejson.dump(featuresets[1], t)
t.close()

num_folds = 10
featuresets = featuresets[:len(featuresets)- len(featuresets)%10]
subset_size = math.trunc(len(featuresets) / num_folds)
#print(len(featuresets))
#print(subset_size)

for i in range(num_folds):
    testing_this_round = featuresets[i*subset_size : (i+1)*subset_size]
    training_this_round = featuresets[:i*subset_size] + featuresets[(i+1)*subset_size:]

    
    #salvo 7 modelli, ma uso solo 5 di questi
    classifier = nltk.NaiveBayesClassifier.train(training_this_round)
    print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_this_round))*100)

    save_classifier = open(str(i)+"originalnaivebayes5k.pickle"  ,"wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()
    
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_this_round)
    print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_this_round))*100)

    save_classifier = open(str(i)+"MNB_classifier5k.pickle","wb")
    pickle.dump(MNB_classifier, save_classifier)
    save_classifier.close()

    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_this_round)
    print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_this_round))*100)

    save_classifier = open(str(i)+"BernoulliNB_classifier5k.pickle","wb")
    pickle.dump(BernoulliNB_classifier, save_classifier)
    save_classifier.close()

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_this_round)
    print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_this_round))*100)

    save_classifier = open(str(i)+"LogisticRegression_classifier5k.pickle","wb")
    pickle.dump(LogisticRegression_classifier, save_classifier)
    save_classifier.close()


    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_this_round)
    print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_this_round))*100)

    save_classifier = open(str(i)+"LinearSVC_classifier5k.pickle","wb")
    pickle.dump(LinearSVC_classifier, save_classifier)
    save_classifier.close()


    ##NuSVC_classifier = SklearnClassifier(NuSVC())
    ##NuSVC_classifier.train(training_set)
    ##print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


    SGDC_classifier = SklearnClassifier(SGDClassifier())
    SGDC_classifier.train(training_this_round)
    print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_this_round)*100)

    save_classifier = open(str(i)+"SGDC_classifier5k.pickle","wb")
    pickle.dump(SGDC_classifier, save_classifier)
    save_classifier.close()

'''