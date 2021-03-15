#!/usr/bin/env python
# coding: utf-8

#------------------------------------------------------------------------------
#Set Seed
#------------------------------------------------------------------------------

import random

random.seed(2020)
print(random.random())

#------------------------------------------------------------------------------
#Detect number of cores on computer (needed for multicore processing to speed up code)
#------------------------------------------------------------------------------

import psutil
psutil.cpu_count()
psutil.cpu_count(logical=False)  # Ignoring virtual cores

#------------------------------------------------------------------------------
#Set working directory
#------------------------------------------------------------------------------

import os
os.chdir("/Users/anavirshermon/Dropbox (Kenan-Flagler)/Drones/Drones 2.0/Empirical Analysis/Newspaper Analysis/MAR9")

#------------------------------------------------------------------------------
#Import relevant packages
#------------------------------------------------------------------------------

import pandas as pd
import os
from pprint import pprint
import csv

from dateutil.parser import parse
from pandas import DataFrame

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore

# spacy for lemmatization
import spacy
from nltk.stem.wordnet import WordNetLemmatizer

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

#------------------------------------------------------------------------------
#Prepare stopwords
#------------------------------------------------------------------------------

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
#stop_words.extend(['using', 'publication', 'ltd', 'elsevier', 'reserved', 'rights'])

#------------------------------------------------------------------------------
#Import newspaper articles
#------------------------------------------------------------------------------
import glob
import docx

#define blank list
words = []

#define function to import USAT, NYT, WaPo
def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:
                if "Bibliography" in file:
                    continue
                if("doclist" in file):
                    continue
                if(".docx" in file):
                    r.append(os.path.join(subdir, file))                                                                         
    return r                                                                                                          


#file_list = list_files("/Users/Mihir/Downloads/Sample/")
file_list = list_files("/Users/anavirshermon/Dropbox (Kenan-Flagler)/Newspaper Analysis/Sample/")

#import wsj articles, which are in an excel sheet
wsj      = pd.read_excel("/Users/anavirshermon/Dropbox (Kenan-Flagler)/Newspaper Analysis/Sample/WSJ articles body - Anavir 25JAN2021.xlsx", engine='openpyxl') 
wsj      = wsj[wsj['Content'].notna()]

wsj['year'] = wsj['date'].dt.year #extract year

wsj_1    = wsj[(wsj['year']<=2012)]
wsj_2    = wsj[(wsj['year']>2012) & (wsj['year']<= 2015)]
wsj_3    = wsj[(wsj['year']>=2016)]

wsj_list   = wsj['Content'].tolist()
wsj_1_list = wsj_1['Content'].tolist()
wsj_2_list = wsj_2['Content'].tolist()
wsj_3_list = wsj_3['Content'].tolist()

    
#add words to files list
for i in file_list:
    doc = docx.Document(i)
    words.append([p.text for p in doc.paragraphs]) 
        
#making each word in paragraph formation and removing all words that come before "Body", which tells us when the paragraph begins
for i in range(len(words)):
    words[i] = ' '.join(words[i])
    #words[i] = words[i][words[0].index("Body "):]
    
one = []
two = [] 
three = []

#split up files between dates
for i in words:
    date =parse(i[i.index("Load-Date:") + 11:i.index("End of Document")])
    if(date.year<=2012):
        one.append(i)
    elif(date.year<=2015):
        two.append(i)
    else:
        three.append(i)

words_cleaned = [x.rsplit("Body ")[1] for x in words]
words_cleaned = [x.split("Load-Date")[0] for x in words_cleaned]
words_cleaned = [x for x in words_cleaned if str(x) != 'nan']

one_cleaned = [x.rsplit("Body ")[1] for x in one]
one_cleaned = [x.split("Load-Date")[0] for x in one_cleaned]
one_cleaned = [x for x in one_cleaned if str(x) != 'nan']

two_cleaned = [x.rsplit("Body ")[1] for x in two]
two_cleaned = [x.split("Load-Date")[0] for x in two_cleaned]
two_cleaned = [x for x in two_cleaned if str(x) != 'nan']

three_cleaned = [x.rsplit("Body ")[1] for x in three]
three_cleaned = [x.split("Load-Date")[0] for x in three_cleaned]
three_cleaned = [x for x in three_cleaned if str(x) != 'nan']

for i in wsj_list:
    words_cleaned.append(i)  

for i in wsj_1_list:
    one_cleaned.append(i)
    
for i in wsj_2_list:
    two_cleaned.append(i)
    
for i in wsj_3_list:
    three_cleaned.append(i)
  

del file_list, wsj, wsj_list, wsj_1, wsj_2, wsj_3, wsj_1_list, wsj_2_list, wsj_3_list, i, doc, one, two, three, words

#------------------------------------------------------------------------------
#Drop military related newspaper articles
#------------------------------------------------------------------------------

#Code to eventually use tf-idf or some systematic method to drop military articles
from pandas import DataFrame
import re

#drop articles if they contain these words
# =============================================================================
# drop_words    = {'military', 'strike', 'attack', 'syria', 'iran', 'pakistan'}
# words_cleaned = [i for i in words_cleaned if not any(x in i for x in drop_words)]
# one_cleaned   = [i for i in one_cleaned if not any(x in i for x in drop_words)]
# two_cleaned   = [i for i in two_cleaned if not any(x in i for x in drop_words)]
# two_cleaned = [i for i in three_cleaned if not any(x in i for x in drop_words)]
# 
# =============================================================================

drop_words= {'military', 'strike', 'attack', 'syria', 'yemen', 'pakistan', 'afghanistan', 'taliban', 'pentagon', 'islamic'}
words_cleaned = [i for i in words_cleaned if not any(x in i for x in drop_words)]
one_cleaned   = [i for i in one_cleaned if not any(x in i for x in drop_words)]
one_cleaned   = [i for i in one_cleaned if not any(x in i for x in drop_words)]
three_cleaned = [i for i in three_cleaned if not any(x in i for x in drop_words)]

#convert to lower case
data        = [item.lower() for item in words_cleaned]
data_one    = [item.lower() for item in one_cleaned]
data_two    = [item.lower() for item in two_cleaned]
data_three  = [item.lower() for item in three_cleaned]

#remove new line characters
#data = [re.sub('\s+', ' ', sent) for sent in data]   # Remove new line characters

#tokenize and pre-process the text
def split(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
data        = list(split(data))
data_one    = list(split(data_one))
data_two    = list(split(data_two))
data_three  = list(split(data_three))


#remove numbers, but not words that contain numbers
data       = [[token for token in doc if not token.isnumeric()] for doc in data]
data_one   = [[token for token in doc if not token.isnumeric()] for doc in data_one]
data_two   = [[token for token in doc if not token.isnumeric()] for doc in data_two]
data_three = [[token for token in doc if not token.isnumeric()] for doc in data_three]

#remove words that are only two characters
data       = [[token for token in doc if len(token) > 2] for doc in data]
data_one   = [[token for token in doc if len(token) > 2] for doc in data_one]
data_two   = [[token for token in doc if len(token) > 2] for doc in data_two]
data_three = [[token for token in doc if len(token) > 2] for doc in data_three]

#remove stopwords
def remove_stopwords(words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in words]
data       = remove_stopwords(data)
data_one   = remove_stopwords(data_one)
data_two   = remove_stopwords(data_two)
data_three = remove_stopwords(data_three)

# =============================================================================
# from collections import Counter 
# test = Counter(c for clist in data for c in clist)
# most_occur = test.most_common(150)
# 
# =============================================================================

#create bigrams and trigrams
bigram         = gensim.models.Phrases(data, min_count=5, threshold=10)
trigram        = gensim.models.Phrases(bigram[data], threshold=10)
bigram_mod     = gensim.models.phrases.Phraser(bigram)
trigram_mod    = gensim.models.phrases.Phraser(trigram)

def make_bigrams(texts):
   return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
   return [trigram_mod[bigram_mod[doc]] for doc in texts]

data_bigrams    = make_bigrams(data)
data_trigrams   = make_trigrams(data)

data_one_trigrams   = make_trigrams(data_one)
data_two_trigrams   = make_trigrams(data_two)
data_three_trigrams = make_trigrams(data_three)

#lemmatize words
lemmatizer = WordNetLemmatizer()
data_lemmatized = [[lemmatizer.lemmatize(token) for token in doc] for doc in data_trigrams]

print(data_lemmatized[99:100])

data_one_lemmatized   = [[lemmatizer.lemmatize(token) for token in doc] for doc in data_one_trigrams]
data_two_lemmatized   = [[lemmatizer.lemmatize(token) for token in doc] for doc in data_two_trigrams]
data_three_lemmatized = [[lemmatizer.lemmatize(token) for token in doc] for doc in data_three_trigrams]


# =============================================================================
# #------NEW TEST FOR LEMMATIZATION-------
# # https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
# # https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
# 
# import nltk 
# from nltk.stem import WordNetLemmatizer 
# nltk.download('averaged_perceptron_tagger') 
# from nltk.corpus import wordnet 
#   
# lemmatizer = WordNetLemmatizer() 
# 
# # Define function to lemmatize each word with its POS tag 
#   
# # POS_TAGGER_FUNCTION : TYPE 1 
# def pos_tagger(nltk_tag): 
#     if nltk_tag.startswith('J'): 
#         return wordnet.ADJ 
#     elif nltk_tag.startswith('V'): 
#         return wordnet.VERB 
#     elif nltk_tag.startswith('N'): 
#         return wordnet.NOUN 
#     elif nltk_tag.startswith('R'): 
#         return wordnet.ADV 
#     else:           
#         return None
#     
#     
#   pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))   
#     
# lemmatized_sentence = [] 
# for word, tag in wordnet_tagged: 
#     if tag is None: 
#         # if there is no available tag, append the token as is 
#         lemmatized_sentence.append(word) 
#     else:         
#         # else use the tag to lemmatize the token 
#         lemmatized_sentence.append(lemmatizer.lemmatize(word, tag)) 
# lemmatized_sentence = " ".join(lemmatized_sentence) 
#   
# 
# data_lemmatized = [[lemmatizer.lemmatize(token) for token in doc] for doc in data_trigrams]
# 
# data_lemmatized = []
# 
# #------NEW TEST FOR LEMMATIZATION-------
# =============================================================================


#------------------------------------------------------------------------------
#Create the Dictionary and Corpus for Topic Modeling
#------------------------------------------------------------------------------

#data_lemmatized = data_lemmatized
#data_lemmatized = data_one_lemmatized
data_lemmatized = data_two_lemmatized
#data_lemmatized = data_three_lemmatized

#group = "_pre2013"
group = "_2013-2015"
#group = "_post2016"

#Method A: Bag of words
id2word = gensim.corpora.Dictionary(data_lemmatized)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
# dictionary.filter_extremes(no_below=20, no_above=0.5)
id2word.filter_extremes(no_below=20, no_above=0.75)

#dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000) 
#Filter out tokens that appear in less than 15 documents (absolute number) or more than 0.5 documents (fraction of total corpus size, not absolute number).
#after the above two steps, keep only the first 100000 most frequent tokens.

texts = data_lemmatized
corpus = [id2word.doc2bow(doc) for doc in texts]

#------------------------------------------------------------------------------
#Build LDA model
#------------------------------------------------------------------------------

# =============================================================================
# '''
# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=6, 
#                                            random_state=100,
#                                            update_every=1,
#                                            chunksize=100,
#                                            passes=10,
#                                            alpha='auto',
#                                            per_word_topics=True)
# '''
# 
# lda_model = LdaMulticore(corpus=corpus,
#                         id2word=id2word,
#                         num_topics=7, 
#                         random_state=100,
#                         chunksize=25,
#                         passes=10,
#                         per_word_topics=True,
#                         workers = 5)
# 
# 
# # Print the Keyword in the 18 topics
# print(lda_model.print_topics())
# 
# #Compute Model Perplexity and Coherence Score
# # Compute Perplexity
# print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
# 
# # Compute Coherence Score
# coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ', coherence_lda)
# 
# =============================================================================
#------------------------------------------------------------------------------
#Find optimal LDA model
#------------------------------------------------------------------------------

def compute_coherence_values(dictionary, corpus, texts, limit, start=5, step=1):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        '''
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        '''
        print("This simulation is current processing", num_topics, "topics")
        model = LdaMulticore(corpus=corpus,
                        id2word=id2word,
                        num_topics=num_topics, 
                        random_state=100,
                        chunksize=5,
                        passes=20,
                        per_word_topics=True,
                        workers = 5)

        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=id2word, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence()) 

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=5, limit=25, step=1)

#PLot coherence values 
limit=25; start=5; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

#Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


#------------------------------------------------------------------------------
#Choose optimal LDA model
#------------------------------------------------------------------------------

#optimal_model = model_list[7]
#model_topics = optimal_model.show_topics(formatted=False)
#pprint(optimal_model.print_topics(num_words=5))

#------------------------------------------------------------------------------
#Export results: top words per topics
#------------------------------------------------------------------------------

#Top words per document
#print(optimal_model.print_topics())

for x in range(5, 24):
    
    optimal_model = model_list[x-5]

    fn = "topic_terms" + str(x) + group + ".csv"
    print(("now creating file ") + fn)
    if (os.path.isfile(fn)):
        m = "a"
    else:
        m = "w"
    
    num_topics=x
    # save topic, term, prob data in the file
    with open(fn, m, encoding="utf8", newline='') as csvfile:
        fieldnames = ["topic_id", "term", "prob"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if (m == "w"):
            writer.writeheader()
    
        for topic_id in range(num_topics):
            term_probs = optimal_model.show_topic(topic_id, topn=35)
            for term, prob in term_probs:
                row = {}
                row['topic_id'] = topic_id
                row['prob'] = prob
                row['term'] = term
                writer.writerow(row)


#------------------------------------------------------------------------------
#Export results: topic similarity
#------------------------------------------------------------------------------

topic_over_topic, annotation = optimal_model.diff(optimal_model, annotation=True)

topic_over_topic_speicherpfad = "topic_over_topic_similarity.csv"
pd.DataFrame(topic_over_topic).to_csv(topic_over_topic_speicherpfad, sep=';')































from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = " ".join(sent)
        texts_out.append([token.lemma_ for token in doc])
    return texts_out


import spacy
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
# Form Bigrams
data_words_bigrams = make_trigrams(data_words_nostops)
import spacy
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
# Form Bigrams
data_words_bigrams = make_trigrams(data_words_nostops)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)


import nltk
words = []
#Lemmatize words and preprocess parargraphs for lda analysis
#
for i in range(len(data_words_bigrams)):
    wn = nltk.WordNetLemmatizer()
    words.append([wn.lemmatize(word) for word in data_words_bigrams[i]])



import re
import string
import nltk
stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem.porter import PorterStemmer



from nltk.corpus import wordnet as wn
#define set of noun anv verbs
nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
verbs = {x.name().split('.', 1)[0] for x in wn.all_synsets('v')}



from nltk.corpus import wordnet
dictionary = {}
#create dictionary based on how many times a word appears in each of the files
for i in range(len(words)):
    s = set(words[i])
    for w in s:
        if(w in dictionary):
            dictionary[w]+=1
        else:
            dictionary[w] = 0
#filteer out words than are less than two letters long, appear in 85% of files, and by part of speach
for i in range(len(words)):
    temp = []
    for w in words[i]:
        temp.append(w)
    for w in words[i]:
        if(dictionary[w]>=(len(words) * 0.85) or len(w)<=2 or (w not in nouns) ):
            temp.remove(w)
    words[i] = temp
temp = []






