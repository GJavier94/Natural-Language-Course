
# coding: utf-8

# In[25]:


import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency

import spacy  # For preprocessing


# In[26]:


#lectura del dataset
df = pd.read_csv('./input/simpsons_dataset.csv')
df.shape
df.head()


# In[27]:


#removemos las filas donde no hay dialogos
df.isnull().sum()
df = df.dropna().reset_index(drop=True)
df.isnull().sum()


# In[28]:


#limpieza del dataset 
#para eso usamos spacy 
"""
Free open-source library for NLP.
It features NER, POS tagging, dependency parsing, word vectors and more.
"""
# descargar el paquete en inglés ----> python -m spacy  download en
nlp = spacy.load('en', disable=['ner', 'parser']) # se desabilita ner y parser


# In[29]:


# removemos caracteres no alfabeticos 
#creamos un generador que va  a sustituir caracter no alfabeticos con espacios

def replace_chars(df, column, pattern, replacement):
    text = []
    for row in df[ column]:
        row = re.sub(pattern, replacement, str(row)).lower()
        text.append(row)
    return text

pattern = "[^A-Za-z']+"
column = 'spoken_words'
text_only_alpha = replace_chars(df, column, pattern, ' ')


# In[30]:



#crear objetos tipo doc de spacy usando un generador
    #spicy funciona ve a los textos como  streams  enviandolos por tuberias(pipes)
docs_gen = nlp.pipe(text_only_alpha, batch_size=5000, n_threads=-1)

#los pasamos a una lista
sp_docs = [] 
for sp_doc in docs_gen:
    sp_docs.append(sp_doc)


# In[31]:


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


# In[32]:


from gensim.models.phrases import Phrases, Phraser
sent = [row.split() for row in clean_docs]


print(clean_docs)

phrases = Phrases(sent, min_count=30, progress_per=10000)

bigram = Phraser(phrases)
sentences = bigram[sent]


# In[33]:


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


# In[34]:


w2v_model.build_vocab(sentences, progress_per=10000)
print(w2v_model)


# In[35]:


w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)


# In[36]:


w2v_model.save("word2vec_simpsons.model")


# In[37]:


#tests

print ( w2v_model.wv.similarity("moe", 'tavern') ) 
print ( w2v_model.wv.similarity('bart', 'nelson') )

print ( w2v_model.wv.doesnt_match(["nelson", "bart", "milhouse"]) ) 
print ( w2v_model.wv.doesnt_match(['homer', 'patty', 'selma']) ) 

