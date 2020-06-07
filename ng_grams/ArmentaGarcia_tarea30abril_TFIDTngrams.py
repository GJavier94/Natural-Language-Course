import pandas as pd
import numpy as np

# In[11]:

def computeTF(wordDict, bag):
    tfDict = {}
    bagCount = len(bag)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagCount)
    return tfDict

# In[13]:

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

# In[15]:

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
    for i in range(0, len(gramOneDoc) - 1 ):
        gramtwo.append( gramOneDoc[i] + ' ' + gramOneDoc[i + 1])
    return gramtwo    



def TFIDF(bags, allWords):
    vectorDocs = []
    for bag in bags:
        vectorDocs.append( gramToVec(allwords, bag) )
    tfDocs = []
    for i in range( 0, len(bags) -1  ):
        tfDocs.append( computeTF(vectorDocs[i], bag[i]) )

    idfs = computeIDF(tfDocs)

    tfidf_docs = []

    for i in range(0, leng(bags) -1):
        tfidf_docs.append(  computeTFIDF(tfDocs[i], idfs) )

    return tfidf_docs


inp  = open('input.txt', 'r') 
documents = inp.readlines() 
  

gramOneList = []

    for document in documents:
        gramOneList.append( document.split(' '))


allwords = set()

    for i in range(0 , len(documents) - 1 ):
        allwords.union( set( gramOneList[i] ) )   


tfidf_onegram = TFIDF(gramOneList, allWords)



gramtwoList = []

    for i in range(0 , len(documents) - 1 ):
        gramtwoList.append( doc2gramtwo(gramOneList[i]) )


allwords = set()

    for i in range(0 , len(documents) - 1 )
        allwords.union( set( gramtwoList[i] ) )   


tfidf_twogram = TFIDF(gramtwoList, allWords)



df_onegram = pd.DataFrame(tfidf_onegram)
df_twogram = pd.DataFrame(tfidf_twogram)




print("INPUT\n")

print(documentA)
print(documentB)
print(documentC)


print("\nOUTPUT\n")
print(df_onegram)
print(df_twogram)



vector_doc_A = {}
vector_doc_A.update(tfidf_onegram[0])
vector_doc_A.update(tfidf_twogram[0])


vector_doc_B = {}
vector_doc_B.update(tfidf_onegram[1])
vector_doc_B.update(tfidf_twogram[1])


vector_doc_C = {}
vector_doc_C.update(tfidf_onegram[2])
vector_doc_C.update(tfidf_twogram[2])


A = []
B = []
C = []
for key,value in vector_doc_A.items():
    A.append( value)

for key,value in vector_doc_B.items():
    B.append( value)

for key,value in vector_doc_C.items():
    C.append( value)


table = np.array([A,B,C])

from sklearn.metrics.pairwise import cosine_similarity
outputSim = cosine_similarity(table)
    
print('Sim a,b ' + str( outputSim[0][1]) )
print('Sim a,c ' + str( outputSim[0][2]) )
print('Sim b,c ' + str( outputSim[1][2]) ) 


import os 

base_filename = 'output_TF_IDF.txt'
with open(os.path.join("/home/javier/Desktop", base_filename),'w') as outfile:
    df_onegram.to_string(outfile)
    df_twogram.to_string(outfile)