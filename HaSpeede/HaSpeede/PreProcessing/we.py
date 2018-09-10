import csv
import operator
import re
import treetaggerwrapper 
from pprint import pprint
import polarita as po
import simplejson

def Sorting(lst):
    lst2 = sorted(lst, key=len, reverse = 1)
    return lst2

def splitta(hashtag):
    #Hashtag to split
    #hashtag = "#nelMezzoDelCamminoDiNostraVita"
    hashtag = hashtag.replace("#","")
    hashtag = hashtag.lower()

    lista = []
    listaDaTogliere = []

    #Here we have to load the dictionary in ".csv" format
    spamReader = csv.reader(open('parole.csv', newline=''), delimiter=' ', quotechar='|')
    for row in spamReader:
      if row[0] in hashtag:
        #We do not consider words that have less than 2 letters
        if len(row[0]) > 2:
          lista.extend([row[0]])

    lista = Sorting(lista)

    #This small block removes all the words found from the initial hashtag, thus obtaining the useless string to be removed
    #The words are deleted in decreasing length, so that smaller strings are not contained in the larger ones
    listaDaTogliere = []
    tmpstring = hashtag
    for i in range(0, len(lista)):
      if lista[i] in tmpstring:
        tmpstring = tmpstring.replace(lista[i],"#")
      else:
        listaDaTogliere.extend([lista[i]])
    tmpstring2 = hashtag
    for i in range(0, len(lista)):
      tmpstring2 = tmpstring2.replace(lista[i],"#"+lista[i]+"#")
    hashtag = tmpstring2.replace(tmpstring,"");
    hashtag = hashtag.replace("#","")
    lista = list(set(lista) - set(listaDaTogliere))

    listaDaTogliere = []
    for i in range(0, len(lista)):
      tmpstring = hashtag
      for k in range(0, len(lista)): 
        if i != k:
          tmpstring = tmpstring.replace(lista[k],"")
      if lista[i] not in tmpstring:
        listaDaTogliere.extend([lista[i]])
    lista = list(set(lista) - set(listaDaTogliere))

    #For each word found We find its position within the hashtag to sort the words
    posizioni = {}
    for i in range(0, len(lista)):
      posizioni.update({lista[i]:[hashtag.find(lista[i])]})
    #At the last We sort all the word
    posizioni = sorted(posizioni.items(), key=operator.itemgetter(1))
    for i in range(0, len(lista)):
      lista[i] = posizioni[i][0]

    frasedue = " ".join(lista)

    return frasedue

pat = re.compile(r"#(\w+)")


def richiamaSplit(commento):
    frase = commento
    frasetre = frase
    frasetre = pat.sub(" ", frasetre)

    if "#" in frase:
      frase = pat.findall(frase)
      frase = "#".join(frase)
      #voglio una stringa
      hashtagpulito = splitta(frase)
      #concateno la stringa alla frase originale
      frasetre = frasetre + hashtagpulito

    return frasetre



def splittaAt(hashtag):
    #Hashtag to split
    #hashtag = "#nelMezzoDelCamminoDiNostraVita"
    hashtag = hashtag.replace("@","")
    hashtag = hashtag.lower()

    lista = []
    listaDaTogliere = []

    #Here we have to load the dictionary in ".csv" format
    spamReader = csv.reader(open('parole.csv', newline=''), delimiter=' ', quotechar='|')
    for row in spamReader:
      if row[0] in hashtag:
        #We do not consider words that have less than 2 letters
        if len(row[0]) > 2:
          lista.extend([row[0]])

    lista = Sorting(lista)

    #This small block removes all the words found from the initial hashtag, thus obtaining the useless string to be removed
    #The words are deleted in decreasing length, so that smaller strings are not contained in the larger ones
    listaDaTogliere = []
    tmpstring = hashtag
    for i in range(0, len(lista)):
      if lista[i] in tmpstring:
        tmpstring = tmpstring.replace(lista[i],"@")
      else:
        listaDaTogliere.extend([lista[i]])
    tmpstring2 = hashtag
    for i in range(0, len(lista)):
      tmpstring2 = tmpstring2.replace(lista[i],"@"+lista[i]+"@")
    hashtag = tmpstring2.replace(tmpstring,"");
    hashtag = hashtag.replace("@","")
    lista = list(set(lista) - set(listaDaTogliere))

    listaDaTogliere = []
    for i in range(0, len(lista)):
      tmpstring = hashtag
      for k in range(0, len(lista)): 
        if i != k:
          tmpstring = tmpstring.replace(lista[k],"")
      if lista[i] not in tmpstring:
        listaDaTogliere.extend([lista[i]])
    lista = list(set(lista) - set(listaDaTogliere))

    #For each word found We find its position within the hashtag to sort the words
    posizioni = {}
    for i in range(0, len(lista)):
      posizioni.update({lista[i]:[hashtag.find(lista[i])]})
    #At the last We sort all the word
    posizioni = sorted(posizioni.items(), key=operator.itemgetter(1))
    for i in range(0, len(lista)):
      lista[i] = posizioni[i][0]

    frasedue = " ".join(lista)

    return frasedue

pat2 = re.compile(r"@(\w+)")


def richiamaSplitAt(commento):
    frase = commento
    frasetre = frase
    frasetre = pat2.sub(" ", frasetre)

    if "@" in frase:
      frase = pat2.findall(frase)
      frase = "@".join(frase)
      #voglio una stringa
      hashtagpulito = splittaAt(frase)
      #concateno la stringa alla frase originale
      frasetre = frasetre + hashtagpulito

    return frasetre


tagger = treetaggerwrapper.TreeTagger(TAGLANG="it")

def lemmatizza(frase):
  frasefinale = ""
  try:
    b = frase.split()
    frasef = []
    for parole in b:
      tags = tagger.tag_text(parole)
      pos = treetaggerwrapper.make_tags(tags)
      #pprint(pos)
      for w in pos:
        if parole == "vai":
          frasef.append("andare")
        else:
          a = w[2].lower()
          if "|" not in a:
            frasef.append(w[2].lower())
          else:
            a = a.replace("|", "I")
            a = re.sub(r'.*I', '', a)
            frasef.append(a)

      frasefinale = " ".join(frasef)
  except:
    pass

  return frasefinale


def percentualeErrori(frase):
  cnt = 0
  lista = frase.split()
  spamReader = csv.reader(open('paroletxt.txt', newline=''), delimiter=' ', quotechar='|')
  for row in spamReader:
      for i in range(0,len(lista)):
        if row[0] == lista[i]:
          #print(row[0])
          cnt = cnt + 1
  
  percentualeErrore = ((len(lista)-cnt)/len(lista))
  percentualeErrore = round(percentualeErrore, 4)
  #percentualeErrore = percentualeErrore*10000      
  #percentualeErrore = int(percentualeErrore)
  b = str(percentualeErrore)
  
  return b.split()

from difflib import SequenceMatcher
from operator import itemgetter

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


#spamReader = csv.reader(open('parole_plus_parolacce.csv', newline=''), delimiter=' ', quotechar='|')
#parole = []
#for row in spamReader:
#    parole.append(row)

def similaritaParola(frase, parole):
  paroleSbagliate = []
  frase = " " + frase + " "
  lista = frase.split()
  similarity = []
  frasefinale = []
  item2 = ""
  
  #for row in spamReader:
  #  parole.append(row)

  flat_list = []
  for sublist in parole:
    for item in sublist:
        flat_list.append(item)
      
  paroleA = []
  paroleB = []
  paroleC = []
  paroleD = []
  paroleE = []
  paroleF = []
  paroleG = []
  paroleH = []
  paroleI = []
  paroleJ = []
  paroleK = []
  paroleL = []
  paroleM = []
  paroleN = []
  paroleO = []
  paroleP = []
  paroleQ = []
  paroleR = []
  paroleS = []
  paroleT = []
  paroleU = []
  paroleV = []
  paroleW = []
  paroleX = []
  paroleY = []
  paroleZ = []

  for f in flat_list:
    if f[0] == 'a':
      paroleA.append(f)
    elif f[0] == 'b':
      paroleB.append(f)
    elif f[0] == 'c':
      paroleC.append(f)
    elif f[0] == 'd':
      paroleD.append(f)
    elif f[0] == 'e':
      paroleE.append(f)
    elif f[0] == 'f':
      paroleF.append(f)
    elif f[0] == 'g':
      paroleG.append(f)
    elif f[0] == 'h':
      paroleH.append(f)
    elif f[0] == 'i':
      paroleI.append(f)
    elif f[0] == 'j':
      paroleJ.append(f)
    elif f[0] == 'k':
      paroleK.append(f)
    elif f[0] == 'l':
      paroleL.append(f)
    elif f[0] == 'm':
      paroleM.append(f)
    elif f[0] == 'n':
      paroleN.append(f)
    elif f[0] == 'o':
      paroleO.append(f)
    elif f[0] == 'p':
      paroleP.append(f)
    elif f[0] == 'q':
      paroleQ.append(f)
    elif f[0] == 'r':
      paroleR.append(f)
    elif f[0] == 's':
      paroleS.append(f)
    elif f[0] == 't':
      paroleT.append(f)
    elif f[0] == 'u':
      paroleU.append(f)
    elif f[0] == 'v':
      paroleV.append(f)
    elif f[0] == 'w':
      paroleW.append(f)
    elif f[0] == 'x':
      paroleX.append(f)
    elif f[0] == 'y':
      paroleY.append(f)
    elif f[0] == 'z':
      paroleZ.append(f)

  
  
  for l in lista:
    letteraIniziale = l[0]
    if letteraIniziale == 'a':
      flat_list = paroleA
    elif letteraIniziale == 'b':
      flat_list = paroleB
    elif letteraIniziale == 'c':
      flat_list = paroleC
    elif letteraIniziale == 'd':
      flat_list = paroleD
    elif letteraIniziale == 'e':
      flat_list = paroleE
    elif letteraIniziale == 'f':
      flat_list = paroleF
    elif letteraIniziale == 'g':
      flat_list = paroleG
    elif letteraIniziale == 'h':
      flat_list = paroleH
    elif letteraIniziale == 'i':
      flat_list = paroleI
    elif letteraIniziale == 'j':
      flat_list = paroleJ
    elif letteraIniziale == 'k':
      flat_list = paroleK
    elif letteraIniziale == 'l':
      flat_list = paroleL
    elif letteraIniziale == 'm':
      flat_list = paroleM
    elif letteraIniziale == 'n':
      flat_list = paroleN
    elif letteraIniziale == 'o':
      flat_list = paroleO
    elif letteraIniziale == 'p':
      flat_list = paroleP
    elif letteraIniziale == 'q':
      flat_list = paroleQ
    elif letteraIniziale == 'r':
      flat_list = paroleR
    elif letteraIniziale == 's':
      flat_list = paroleS
    elif letteraIniziale == 't':
      flat_list = paroleT
    elif letteraIniziale == 'u':
      flat_list = paroleU
    elif letteraIniziale == 'v':
      flat_list = paroleV
    elif letteraIniziale == 'w':
      flat_list = paroleW
    elif letteraIniziale == 'x':
      flat_list = paroleX
    elif letteraIniziale == 'y':
      flat_list = paroleY
    elif letteraIniziale == 'z':
      flat_list = paroleZ

    if l not in flat_list:
      print("cerco")
      paroleSbagliate.append(l)
      maxSimile = 0
      for p in flat_list:
        simile = similar(p,l)
        if simile >= 0.8 or simile >= maxSimile:
          maxSimile = simile
          item2 = p
      if item2 == "":
          frasefinale.append(l) 
      else:
          frasefinale.append(item2)
          item2 = ""
    else:
      frasefinale.append(l)
    

  frasedue = " ".join(frasefinale)
  #print(frasedue)

  return frasedue


  #for p in paroleSbagliate:
  #  maxSimile = 0
  #  for l in flat_list:
  #    simile = similar(p,l)
  #    if simile >= 0.8:
  #      maxSimile = simile
  #      item = (l, simile)
  #      similarity.append(item)

  #print(similarity)
#print(similar("caprone","pastorizia"))
#print(similaritaParola("MAMMA MIA PESE COME IL BACCALÀ COI PORRI!!!!!!!!!!! MA CHI LE HA VOTATE?  LA MALPEZZI SEMBRA LA STRGA DI NERANEVE ( visto chi difende) LA CASTALDINI INCLASSIFICABILE!!  NCD  (nuovo centro distruzione)"))
  #print(max(similarity, key=operator.itemgetter(1)))
      

#print(percentualeErrori(" ho cambiato canale    pd ncd fanno schifo"))

#print(percentualeErrori("piu di te"))
#print(percentualeErrori("non essere malo visto nido costa quanto rato mutuo più"))
  #return percentualeErrore
array = ['ino ','ina ','ini ','ine ','one ','ona ','oni ', 'accio ', 'accia ', 'acci ', 'acce ','otto ','otta ','otte ', 'otti ' , 'issimo ', 'issimi ', 'issime ', 'issima ']
 
def pulisciCommento(frase):
  #s = "spassosissimo stupidissima schifosino costosissima vecchiaccia giovanotto"
  s1 = frase
  cnt = 0
  lista = frase.split()
  lista1 = lista.copy()

  spamReader = csv.reader(open('parole2.csv', newline=''), delimiter=';', quotechar='|')
  
  #print(lista1)

  for row in spamReader:
    for i in range(0, len(lista)):
      if row[0] == lista1[i]:
          lista1[i] = ""
  #print("AAAA ",lista1)

  lista1 = [suit + " " for suit in lista1]

  for i in range(0, len(lista)):
    for j in range(0,len(array)):
      if array[j] in lista1[i]:
        lista1[i] = lista1[i].replace(array[j],lista[i][-1])
        lista[i] = lista1[i]

  frase = " ".join(lista)

  return frase

def trovaParolacce(frase):
  cnt = 0
  c = 0
  spamReader = csv.reader(open('parolacce.csv', newline=''), delimiter=';', quotechar='|')
  for row in spamReader:
    if " "+row[0] in frase:
      cnt = cnt + 1
  if len(frase) != 0:
    c = (cnt/len(frase))
    c = round(c,4)
    #c = int(c)
    #c = c*10000
    
  b = str(c)
  return b.split()


def contaParolacce(frase):
  cnt = 0
  c = 0
  spamReader = csv.reader(open('parolacce.csv', newline=''), delimiter=';', quotechar='|')
  for row in spamReader:
    if " "+row[0] in frase:
      cnt = cnt + 1

  b = str(cnt)
  return b.split()

from textblob import TextBlob

def polaritaFrase(frase):
  if len(frase) != 0:
    try:
      fraseTradotta = str(TextBlob(frase).translate(to="en"))
      #print(fraseTradotta)
    except:
      print("non ho tradotto")
      return [0]
  else:
    return [0]
  #print(fraseTradotta)
  score = po.calcolaPolarita(fraseTradotta)
  score = round(score,4)
  #score = score*10000
  scorestr = str(score)

  return scorestr.split()

def polaritaTextBlob(frase):
  if len(frase) != 0:
    try:
      fraseTradotta = str(TextBlob(frase).translate(to="en"))
      #print(fraseTradotta)
    except:
      print("non ho tradotto")
      return [0]
  else:
    return [0]
  testimonial = TextBlob(fraseTradotta)
  score = testimonial.sentiment.polarity
  scorestr = str(score)

  return scorestr.split()

def soggettivitaTextBlob(frase):
  if len(frase) != 0:
    try:
      fraseTradotta = str(TextBlob(frase).translate(to="en"))
      #print(fraseTradotta)
    except:
      print("non ho tradotto")
      return [0]
  else:
    return [0]
  testimonial = TextBlob(fraseTradotta)
  score = testimonial.sentiment.subjectivity
  scorestr = str(score)

  return scorestr.split()

#numero di punti interrogativi all'interno della frase
def numeroInterrEscl(frase):
  n = frase.count('?') + frase.count('!')
  s = str(n)
  return s.split()

#cerca il numero di parole in caps lock
def cercaCapsLock(frase):
  parole = frase.split()
  cnt3 = 0
  for parola in parole:
    cnt = len(parola)
    cnt2 = sum(1 for c in parola if c.isupper())
    if cnt == cnt2:
      cnt3 += 1
  percentuale = (cnt3/len(parole))
  percentuale = round(percentuale,4)
  #percentuale = percentuale*10000
  #percentuale = int(percentuale)
  #print(percentuale)
  s = str(percentuale)
  return s.split()

#trova la lunghezza della frase
def lunghezzaFrase(frase):
  lenght = str(len(frase.split()))
  return lenght.split()

#conta le punteggiature nella frase
def numeroPunteggiatura(frase):
  n = frase.count('.') + frase.count(',')
  s = str(n)
  return s.split()



def convertiParolacce(frase):
  parolacce = ["cazzo","puttana","troia", "stronzo", "cretino","cagare","culo","merda"]
  lista = frase.split()
  separatore = ""
  frase = " " + frase + " "
  try:
    for i in lista:
      separatore = ""
      if (i[0].isalpha() and i[-1].isalpha()):
        l = list(i[1:-1])
        for k in l:
          if (not k.isalpha()) or "x" in k:
            separatore = separatore + k
        if(separatore != ""):
          l = i.split(separatore)
          for p in parolacce:
            if p[:len(l[0])] == l[0] and p[-len(l[1]):] == l[1]:
              frase = frase.replace((" "+i+" "), " "+p+" ")
  except:
    pass
  return frase


def rimuoviRisate(frase):
  #s = "eheheheh ihihihi ohohoho uhuhuhuhu ahahahahahah hahahaha hehehehe ooo cciaociao"
  lista = frase.split()
  vocali = ["a", "e", "i", "o", "u"]

  for i in lista:
    for k in vocali:
      if ("h" in i) and (len(i) >= 4):
        if (len(i) - 2) <= (i.count(k) + i.count('h')):
          frase = frase.replace(i, "")

  return frase

import gzip
import gensim 
import logging
import csv
import time

def calcolaSimilarita(frase):
    parole = []
    parole2 = []
    parole3 = []
    #parole 4 lo costruisco con i commenti puliti per allenare il modello
    parole4 = []
    #parole 5 contiene le frasi con cui viene calcolata la similarita
    parole5 = []
    with open("haspeede_FB-train.tsv",encoding="utf8") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')

        for row in rd:
          row.pop(0)
          row.pop(1)
          row[0] = row[0].split()
          parole.append(row)
    
    with open("myFile1.csv",encoding="utf8") as fd:
        rd = csv.reader(fd, delimiter=",", quotechar='"')

        for row in rd:
            riga = []
            riga.append(row[1].split())
            parole4.append(riga)
            
    with open("haspeede_FB-train.tsv",encoding="utf8") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')

        for row in rd:
            riga = []
            riga.append(row[1])
            riga.append(row[2])
            parole3.append(riga)
    
    with open("myFile1.csv",encoding="utf8") as fd:
        rd = csv.reader(fd, delimiter=",", quotechar='"')

        for row in rd:
            riga = []
            riga.append(row[1])
            riga.append(row[2])
            parole5.append(riga)

    #con parole4 creo flat_list con il quale alleno il word2vec model
    flat_list = []
    for sublist in parole4:
      for item in sublist:
        flat_list.append(item) 

    #model = gensim.models.Word2Vec(flat_list, size=1500, window=10,min_count=2,workers=4)
 #   #model.train(flat_list, total_examples=len((flat_list)), epochs=20)
##
    #model.save("word2vec.model")
    #model = gensim.models.Word2Vec.load('it.bin')
    model = gensim.models.Word2Vec.load("word2vec.model")

    fraseSimile = []
    valoreMassimo = 0
    for p in parole5:
      distance = model.wmdistance(p[0], frase)
      #print(distance)
      if distance <= 0.1:
        #print ('distance = %.3f' % distance)
        valoreMassimo = distance
        fraseSimile.append(p)
    
    if fraseSimile != []:
        fraseSimile.pop(0)
    #for f in fraseSimile:
        #print(f[0],f[1])
    somma = 0
    lunghezza = len(fraseSimile)
    for f in fraseSimile:
        if f[1] == '1':
            somma += 1

    #print("\n")
    #print("per la frase",frase,"\n")
    #print("sono stati trovati",lunghezza,"commenti molto simili semanticamente")
    #print("su ",lunghezza,"commenti simili sotto il livello 0.1 di similarità semantica")
    #print(somma,"di questi commenti sono negativi")
    #if somma/lunghezza > 0.5:
    #    print("quindi è possibile che sia un commento più negativo che positivo")
    #elif somma/lunghezza < 0.5:
    #    print("quindi è possibile che sia un commento più positivo che negativo")
    #elif somma/lunghezza == 0.5:
    #    print("non mi sbilancio")
    
    if somma != 0:
      b = (somma/lunghezza)
      b = round(b,4)
      #b = b*10000
      #b = int(b)
      a = str(b)
    else:
      a = "0"

    return a.split()

def countSentences(comment):
    sentences = re.split(r'[.!?]+', comment)
    ccSentences = len(sentences)
    if (sentences[len(sentences)-1] == ''):
        ccSentences -=1
    if (sentences[0] == ''):
        ccSentences -=1
    b = str(ccSentences)
    return b.split()

import regex
import emoji

def convertiFaccine(string):

  def split_count(text):
      emoji_counter = 0
      data = regex.findall(r'\X', text)
      for word in data:
          if any(char in emoji.UNICODE_EMOJI for char in word):
              emoji_counter += 1
              text = text.replace(word, '') 

      words_counter = len(text.split())

      return emoji_counter

  emoji_pattern = re.compile("["
          u"\U0001F600-\U0001F64F"  # emoticons
          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
          u"\U0001F680-\U0001F6FF"  # transport & map symbols
          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                             "]+", flags=re.UNICODE)

  #string = "lorenzo ferri è 💩 💩"
  stringa = string
  counter = split_count(string)
  for i in range(0,counter):
    stringa = emoji_pattern.sub(" " , string) 

  tmp = []
  for ch in string:
    if ch not in stringa:
      tmp.append(ch)

  frasedatradurre = "".join(tmp)
  frasedatradurre = emoji.demojize(frasedatradurre)
  frasedatradurre = frasedatradurre.replace(":"," ")
  #fraseTradotta = str(TextBlob(frasedatradurre).translate(to="it"))

  fraseRifatta = frasedatradurre

  return fraseRifatta


#print(convertiFaccine("Complimenti a chi sostiene ancora questa politica marcia. 👏👏👏"))