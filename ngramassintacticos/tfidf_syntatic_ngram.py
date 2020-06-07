import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import codecs

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

   





##TFIDF for 1-gram
encod = 'utf-8'

sngramList = []

file = codecs.open ("output0.txt" ,  "rU", encoding = encod)
sngramList.append( [line.rstrip() for line in file.readlines() ]) 
file = codecs.open ("output1.txt" ,  "rU", encoding = encod)
sngramList.append( [line.rstrip() for line in file.readlines() ]) 
file = codecs.open ("output2.txt" ,  "rU", encoding = encod)
sngramList.append( [line.rstrip() for line in file.readlines() ]) 




set_sn_tri_grams = set()
for i in range(0 , len(sngramList) ):
    set_sn_tri_grams = set_sn_tri_grams.union( set( sngramList[i] ) )   

tfidf_sn_trigram = TFIDF(sngramList, set_sn_tri_grams)

df_sn_trigram = pd.DataFrame(tfidf_sn_trigram)


	


##printin results




print("\nOUTPUT\n")
printDataFrames([df_sn_trigram])







print("Cosine similarity ONLY TRIGRAM-WORD")
table = np.array(df_sn_trigram)
outputSim = cosine_similarity(table)
print_cos_sim_docs(outputSim)

df_sn_trigram.to_csv('df_sn_trigram.csv')



