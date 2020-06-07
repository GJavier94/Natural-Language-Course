import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

def computeTF(wordDict, bag):
    tfDict = {}
    bagCount = len(bag)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagCount)        
    return tfDict

def computeIDF(documents):
    import math
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTFIDF(tfbag, idfs):
    tfidf = {}
    for word, val in tfbag.items():
        tfidf[word] = val * idfs[word]
    return tfidf


def gramToVec(allgrams, gramDoc ):    
    vectorDoc = dict.fromkeys(allgrams, 0)
    for gram in gramDoc:
        vectorDoc[gram] += 1    
    return vectorDoc


def doc2gramtwo(gramOneDoc):
    gramtwo = []
    for i in range(0, len(gramOneDoc) -1 ):
        gramtwo.append( gramOneDoc[i] + ' ' + gramOneDoc[i + 1])
    return gramtwo    



def TFIDF(bags, allwords):
    vectorDocs = []
    for bag in bags:
        vectorDocs.append(gramToVec(allwords, bag))
    
    tfDocs = []
    for i in range( 0, len(bags) ):
        tfDocs.append( computeTF(vectorDocs[i], bags[i]) )

    idfs = computeIDF(tfDocs)

    tfidf_docs = []

    for i in range(0, len(bags)):
        tfidf_docs.append(  computeTFIDF(tfDocs[i], idfs) )

    return tfidf_docs

def printDocuments( documents):
    for document in documents:
        print(document)

def printDataFrames(dataFrames):
    for df in dataFrames:
        print(df)



def print_cos_sim_docs( outputSim ):   
    for i in range(0, outputSim.shape[0] ):
        for j in range(i + 1, outputSim.shape[1]):
            print('Sim ' + str((i+1)) + ',' + str( (j+1)) + ': ' +str(outputSim[i][j]))

   





inp  = open('input.txt', 'r') 
documents = inp.readlines() 




##TFIDF for 1-gram

gramoneList = []
for i in range(0 , len(documents) ):
    gramoneList.append( documents[i].rstrip().split(' ') )



set_one_grams = set()
for i in range(0 , len(gramoneList) ):
    set_one_grams = set_one_grams.union( set( gramoneList[i] ) )   


tfidf_onegram = TFIDF(gramoneList, set_one_grams)
df_onegram = pd.DataFrame(tfidf_onegram)
	


##TFIDF for 2-gram

gramtwoList = []

for i in range(0 , len(documents)):
    gramtwoList.append( doc2gramtwo(gramoneList[i]) )
set_two_grams = set()
for i in range(0 , len(documents) ):
    set_two_grams = set_two_grams.union( set( gramtwoList[i] ) )   
tfidf_twogram = TFIDF(gramtwoList, set_two_grams)
df_twogram = pd.DataFrame(tfidf_twogram)


## trigrams of words

trigramsList = []

for document in documents: 
	trigram = []
	for j in range(0, len(document) - 2):
		trigram.append(document[j:j+3])
	trigramsList.append(trigram)
set_tri_grams_words = set()
for i in range(0 , len(documents) ):
    set_tri_grams_words = set_tri_grams_words.union( set( trigramsList[i] ) )   
tfidf_trigram_word = TFIDF(trigramsList, set_tri_grams_words)
df_trigram_word = pd.DataFrame(tfidf_trigram_word)


##printin results

print("INPUT\n")
printDocuments(documents)


print("\nOUTPUT\n")
printDataFrames([df_onegram, df_twogram , df_trigram_word])


vectors = []

for i in range(0, len(documents) ):
    dic = {}
    dic.update(tfidf_onegram[i])
    dic.update(tfidf_twogram[i])
    dic.update(tfidf_trigram_word[i]) 
    vectors.append([ value for value in dic.values()] )


print("Cosine similarity ONLY TRIGRAM-WORD")
table = np.array(df_trigram_word)
outputSim = cosine_similarity(table)
print_cos_sim_docs(outputSim)



print("Cosine similarity  FEATURES: ONEGRAM,TWOGRAM AND TRIGRAM-WORD")
table = np.array(vectors)
outputSim = cosine_similarity(table)
print_cos_sim_docs(outputSim)


df_onegram.to_csv('df_onegram.csv')
df_twogram.to_csv('df_twogram.csv')
df_trigram_word.to_csv('df_trigram_word.csv')




