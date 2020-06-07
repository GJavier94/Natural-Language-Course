
# coding: utf-8

# In[24]:


import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency

import spacy  # For preprocessing


# In[25]:


filename = "./input/bbt/datasetTBBT.csv"
f = open(filename, "r")  
corpus = f.readlines()
corpus = [ line.rstrip("\n") for line in corpus]


# In[26]:


print(len(corpus))


# In[27]:


#limpieza del dataset 
#para eso usamos spacy 
"""
Free open-source library for NLP.
It features NER, POS tagging, dependency parsing, word vectors and more.
"""
# descargar el paquete en inglés ----> python -m spacy  download en
nlp = spacy.load('en', disable=['ner', 'parser']) # se desabilita ner y parser


# In[28]:


# removemos caracteres no alfabeticos 
#creamos un generador que va  a sustituir caracter no alfabeticos con espacios

def replace_chars( data, pattern, replacement):
    text = []
    for row in data:
        row = re.sub(pattern, replacement, str(row)).lower()
        text.append(row)
    return text

pattern = "[^A-Za-z’]+"
replacement = " "
text_only_alpha = replace_chars(corpus, pattern, replacement)
print(text_only_alpha)


# In[29]:



#crear objetos tipo doc de spacy usando un generador
    #spicy funciona ve a los textos como  streams  enviandolos por tuberias(pipes)
docs_gen = nlp.pipe(text_only_alpha, batch_size=5000, n_threads=-1)

#los pasamos a una lista
sp_docs = [] 
for sp_doc in docs_gen:
    sp_docs.append(sp_doc)


# In[30]:


#removemos stop words y lemmatizamos
def remove_stop_words_lemmatize(sp_docs):
    clean_docs = []
    for doc in sp_docs:
        clean_doc = []
        for token in doc:
            if not token.is_stop:
                clean_doc.append(token.lemma_ )
        if len(clean_doc) > 2: #los textos con solo dos tokens no valen la pena
            clean_docs.append( ' '.join(clean_doc) ) 
    return clean_docs

clean_docs = remove_stop_words_lemmatize(sp_docs)
print(clean_docs)


# In[31]:


from gensim.models.phrases import Phrases, Phraser
sent = [row.split() for row in clean_docs]


print(sent)

phrases = Phrases(sent, min_count=30, progress_per=10000)

bigram = Phraser(phrases)
sentences = bigram[sent]


# In[32]:


import multiprocessing

from gensim.models import Word2Vec
cores = multiprocessing.cpu_count() # Count the number of cores in a computer


w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)


# In[33]:


w2v_model.build_vocab(sentences, progress_per=10000)
print(w2v_model)


# In[34]:


w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)


# In[42]:


w2v_model.save("word2vec_tbt.model")


# In[41]:


print( w2v_model.wv.similarity("leonard", 'sheldon') ) 
print( w2v_model.wv.similarity("black", 'white') ) 
print( w2v_model.wv.similarity("penny", 'bernadette') ) 
print( w2v_model.wv.similarity("geology", 'physics') ) 

