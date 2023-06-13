#!/usr/bin/env python
# coding: utf-8

# In[4]:


import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize 
nltk.download('reuters')
from nltk.corpus import reuters
import random
import re
import time
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import math
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool


# In[5]:


import multiprocessing

cores = multiprocessing.cpu_count()
cores  


# In[10]:


list_of_words = [j for i in reuters.sents() for j in i]


# In[11]:


nltk.download('stopwords')
from nltk.corpus import stopwords
st = stopwords.words('english')
list_of_words1= [i.lower() for i in list_of_words if i.isalpha()== True]
new_list_of_words = [i for i in list_of_words1 if i not in st]

len(new_list_of_words)


# In[12]:


dict_ = {}
for i in new_list_of_words:
    dict_[i] = dict_.get(i,0)+1      #Creating a dictionary from the new list of words.
dict_key_list = list(dict_.keys())
#print(dict_)
print(len(dict_))


# In[13]:


def doc_creation(doc_key_len,number_of_doc = 2):  #This function is used to create the documents from the dataset.
    list_doc = []

    for i in range(number_of_doc):
        rand_list_numbersi = random.sample(list(range(len(dict_key_list))), doc_key_len)  
        doc_i = {dict_key_list[j]:dict_[dict_key_list[j]] for j in rand_list_numbersi}
        list_doc.append(doc_i)

    return list_doc


# In[14]:


# created a list of 16 documents with each dictionary has a bag size of 1000
all_pair_doc_list = doc_creation(1000,16)


# In[15]:




# Defining the mapper function for the map-reduce implementation
def Mapper_all_pair_jaccard(doc_):
  list_output = []
  (i,j,doc_i,doc_j) = doc_
  

  similarity_ = jaccard(doc_i,doc_j)
  list_output.append(((i,j),similarity_))
  return list_output


# In[16]:


def map_reduce_parallel(inputs,mapper,reducer,mapprocesses=1,reduceprocesses=1):
    
    collector=defaultdict(list)  #this dictionary is where we will store intermediate results
                                 #it will map keys to lists of values (default value of a list is [])
                                 #in a real system, this would be stored in individual files at the map nodes
                                 #and then transferred to the reduce nodes
    
    mappool = Pool(processes=mapprocesses)
    #map stage
    
    mapresults=mappool.map(mapper,inputs)
    mappool.close()
    
    for mapresult in mapresults:
        for (key, value) in mapresult:     #pass each input to the mapper function and receive back each key,value pair yielded
            collector[key].append(value)     #append the value to the list for that key in the intermediate store
            
    #reduce stage 
    outputs=[]
    reducepool = Pool(processes=reduceprocesses)
    
    reduceresults=reducepool.map(reducer,collector.items())
    reducepool.close()
    for reduceresult in reduceresults:
        outputs+=reduceresult
   
    return outputs


# In[17]:


#Defining the reducer function for the map-reduce implementation
def reducer_all_pair(item):
  (keys,values) = item
  output_reducer = [(keys,values)]
  return (output_reducer)


# In[18]:


def mapreduce_allpair_jaccard (all_pair_docc_list):
  mapper_iterator = []
  for i in range(len(all_pair_docc_list)):
    for j in range (len(all_pair_docc_list)):
      mapper_iterator.append((i,j,all_pair_docc_list[i],all_pair_docc_list[j]))

  return map_reduce_parallel(mapper_iterator,Mapper_all_pair_jaccard,reducer_all_pair)


# In[ ]:


#stores result as a list of tuples with doument index and corresponding similarity measure
result_jaccard = mapreduce_allpair_jaccard(all_pair_doc_list)


# In[ ]:


print(result_jaccard)


# In[ ]:


jaccard_values =all_pair_comparison(all_pair_doc_list,method='jaccard')
document_index = [i[0] for i in result_jaccard]
jaccard_similarity_mr = [i[1] for i in result_jaccard]


# In[ ]:


data = {'document index': document_index, 'jaccard similarity using mapreduce': jaccard_similarity_mr, 'jaccard similarity without mapreduce': jaccard_values }
df_jacc = pd.DataFrame(data)
df_jacc


# In[ ]:


#defining mapper function for cosine similarity 
def Mapper_all_pair_cosine(doc_):
  list_output = []
  (i,j,doc_i,doc_j) = doc_
  

  similarity_ = cosinesim_sparse(doc_i,doc_j)
  list_output.append(((i,j),similarity_))
  return list_output


# In[ ]:


#defining the main mapreduce function to be called for computing cosine similarity
def mapreduce_allpair_cosine (all_pair_docc_list):
  mapper_iterator = []
  for i in range(len(all_pair_docc_list)):
    for j in range (len(all_pair_docc_list)):
      mapper_iterator.append((i,j,all_pair_docc_list[i],all_pair_docc_list[j]))

  return map_reduce_parallel(mapper_iterator,Mapper_all_pair_cosine,reducer_all_pair)


# In[ ]:


#stores result as a list of tuples with doument index and corresponding similarity measure
result_cosine = mapreduce_allpair_cosine(all_pair_doc_list)


# In[ ]:


print(result_cosine) 


# In[ ]:


cosine_values =all_pair_comparison(all_pair_doc_list,method='cosine_sparse')
document_index = [i[0] for i in result_cosine]
cosine_similarity_mr = [i[1] for i in result_cosine]


# In[ ]:


data = {'document index': document_index, 'cosine similarity using mapreduce': cosine_similarity_mr, 'cosine similarity without mapreduce': cosine_values }
df_cos = pd.DataFrame(data)
df_cos


# In[ ]:


#creates list of 100 dictionaries with bag size of 100
all_pair_doc_list_100 = doc_creation(100,100)
all_pair_doc_list_100[0]


# In[ ]:


#the average time taken to run the all pair similarity using MapReduce is found out

a=time_it(mapreduce_allpair_jaccard,all_pair_doc_list_100,number_of_repeats=5)
b=time_it(mapreduce_allpair_cosine,all_pair_doc_list_100,number_of_repeats=5)
print('The average time taken to find all pair similarity of 100 documents with bagsize of 100: \n {} seconds using jaccard \n {} seconds using cosine'.format(a,b))

