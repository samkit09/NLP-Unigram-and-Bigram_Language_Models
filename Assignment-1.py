#!/usr/bin/env python
# coding: utf-8

# ### Part 1 - Computing unsmoothed unigram and bigram probabilities.

# In[1]:


# importing appropriate libraries
from collections import defaultdict
import os
import random
import math
import copy


# In[2]:


# Reading Dataset
pwd = os.getcwd()
corpus = open(pwd+"\\A1_DATASET\\train.txt","r", encoding='utf-8').read()


# In[3]:


corpus1 = copy.copy(corpus)


# In[4]:


# Function that returns list of non-alphabetical words 
def get_non_alpha_words(words):
    t = filter(lambda x: not x.isalpha() and len(x) > 0,words)
    temp = []
    for s in t:
        if s not in temp:
            temp.append(s)
    return temp


# In[5]:


# Bock to check for words with irregular character or symbols
# non_alpha = get_non_alpha_words(corpus.split(" "))
# print("Non-alphabetical character/words = ",len(non_alpha))
# # non_alpha


# In[6]:


def pre_process (ds):
    # defining character to replace
    special_chars = ["\n", "-", '/', "..", '.', "*"]
    sent_ends = [" . ", " ? ", " ! "]
    rep1 = " "
    rep2 = " . <e> <s> "
    #  replacing special character and escape sequence characer with white space character
    for s in special_chars:
        ds = ds.replace(s, rep1)
    #  adding sentence start '<s>' and end '<e>' tags at the start and end of each sentence
    for s in sent_ends:
        ds = ds.replace(s, rep2)
    
    ds = ds.replace("'re ", " are")
    ds = ds.replace("n't ", " not")
    ds = ds.replace("'t ", " not")
    ds = ds.replace("'m ", " am")
    ds = ds.replace("'d ", " would")
    ds = ds.replace("'ve ", " have")
    ds = ds.replace("'ll ", " will")
    ds = ds.replace("'", " ' ")
    
    ds = "<s> " + ds
    ds += " <e> "
    
    # lowering all the character to lowercase so the model don't take words like 'HELLO' and 'hello' as different words
    ds = ds.lower()
    
    return ds


# In[7]:


# def rep_natural_short_forms():
#     sent_ends = [" . ", " ? ", " ! "]
#     rep1 = " "
#     rep2 = " . <e> <s> "
#     #  replacing special character and escape sequence characer with white space character
#     for s in special_chars:
#         ds = ds.replace(s, rep1)


# In[8]:


# preprocessing data
ds = pre_process(corpus)
tokens = ds.split(' ')


# In[9]:


non_alpha = get_non_alpha_words(tokens)
print("Non-alphabetical character/words = ",len(non_alpha))
# non_alpha


# In[10]:


words = [x for x in tokens if len(x)>0]


# In[18]:


# Block for counting words and storing in appropriate data structure

def get_counts(words):
    vocab = []
    unigram = defaultdict(lambda: 0)
    bi_gram = defaultdict(lambda:defaultdict(lambda:0))
    for i in range(len(words)-1):
        if words[i] not in vocab: 
            vocab.append(words[i])
        unigram[words[i]] += 1
        bi_gram[words[i]][words[i+1]] += 1
        
    if len(words) > 0 and words[-1] not in vocab:
        unigram[words[-1]] = 0
    elif len(words) > 0:
        unigram[words[-1]] += 1
        
    return vocab, unigram, bi_gram


# In[19]:


# getting vocab and counts of words for unigram and bigram without unknown word handeling
vocab, unigram, bi_gram = get_counts(words)


# In[130]:


# unigram.index('<unk>')


# In[21]:


# function to handle unknown data
def handle_unigram_unk(unigram, n = 2):
#     print(unigram)
    cp_unk = []
    for k,v in dict(unigram).items():
        if v < n: 
            print(k,v)
            cp_unk.append(k)
    return cp_unk


# In[22]:


# collecting less frequent words
corpus_unk = handle_unigram_unk(unigram)
corpus_unk


# In[23]:


# handeling unknown words by repllacing them with <unk> tag
words_unk = []
for word in words: 
    if word in corpus_unk:
        words_unk.append("<unk>")
    else:
        words_unk.append(word)


# In[24]:


# getting vocab and counts of words for unigram and bigram
vocab_unk, unigram_unk, bi_gram_unk = get_counts(words_unk)


# In[183]:


# def handle_bi_gram_unk(n = 3):
#     bi_gram_unk = defaultdict(lambda:defaultdict(lambda: 0))
#     for key,val in bi_gram.items():
#         temp = defaultdict(lambda: 0)
#         for k,v in val.items():
#             if v < n: 
#                 temp["<unk>"] += v
#             else:
#                 temp[k] = v
#         bi_gram_unk[key] = temp
#     t = 1/unigram_unk['<unk>']
#     bi_gram['<unk>'] = {'<unk>': t}
        
#     return bi_gram_unk


# In[184]:


# vocab_unk


# In[185]:


# bi_gram_unk = handle_bi_gram_unk()


# In[186]:


# bi_gram_unk['<unk>']['<unk>']


# In[25]:


# function to generate sentence of n words
def gen_sent(n = 10):
    words = []
    next_word = "<s>"
#     words.append(next_word)
    while (len(words) < n and next_word != "<e>"):
        next_word = random.choice(list(bi_gram[next_word].keys()))
        words.append(next_word)
#     print(words)
    if words[-1] == '<e>':
        words = words[:-1]
    return " ".join(words)


# In[26]:


# Example of generating a sentence.
gen_sent()


# In[27]:


# function to calculate and store unigram probabilities
def get_unigram_probabilities(counts):
    unigram_probabilities = defaultdict()
    N = sum(list(counts.values()))
    for k,v in counts.items():
        unigram_probabilities[k] = v/N
    return unigram_probabilities


# In[66]:


# function to calculate and store bigram probabilities
def get_bigram_probabilities(counts):
    bigram_probabilities = defaultdict(lambda:defaultdict(lambda:0))
    for k,v in counts.items():
        temp = defaultdict(lambda:0)
        n = sum(list(counts[k].values()))
        for k1,v1 in v.items():
#             print(k1)
            try :
                temp[k1] = v1/counts[k1]
            except:
                temp[k1] = v1/len(vocab_unk)
        bigram_probabilities[k] = temp
    return bigram_probabilities


# In[67]:


# unigram probabilities without unknown word handling
unigram_probabilities = get_unigram_probabilities(unigram)
# unigram_probabilities


# In[68]:


# unigram probabilities with unknown word handling
unigram_probabilities_unk = get_unigram_probabilities(unigram_unk)
# unigram_probabilities_unk


# In[69]:


unigram_probabilities_unk['<unk>']


# In[70]:


unigram['yep']


# In[72]:


# bigram probabilities without unknown word handling
bigram_probabilities = get_bigram_probabilities(bi_gram)
bigram_probabilities


# In[73]:


# bigram probabilities with unknown word handling
bigram_probabilities_unk = get_bigram_probabilities(bi_gram_unk)
bigram_probabilities_unk


# In[74]:


bigram_probabilities_unk['i']["<unk>"]


# ## Part 2 - Applying Smoothing (Laplace and Add-k)

# ### Formula for Add-k smoothing:
# ![image.png](attachment:image.png)

# In[95]:


def get_prob_after_smoothing_uni(counts, K = 1):
    N = sum(list((counts.values())))
    _prob = defaultdict(lambda:(1+K)/(N+len(vocab_unk)*K))
    for k,v in counts.items():
        # Formula for add-k smoothing
        
        _prob[k] = (v + K) / (N + (K*len(vocab_unk)))
    return _prob


# In[98]:


def get_prob_after_smoothing_bi(counts, unigram, K = 1):
    '''
    counts: key value pairs with words as keys and count as values. Used to calculate probabilities for unigrams and bigrams
    '''
    _prob = defaultdict(lambda:defaultdict(lambda:1/len(vocab)))
    for k,v in counts.items():
        n = unigram[k]
        temp = defaultdict(lambda:(K/(n + K*len(vocab_unk))))
        for k1,v1 in v.items():
#            try:
                # Formula for add-k smoothing
            temp[k1] = (v1+K)/(n + K * len(vocab_unk))
#             except:
#                temp[k1] = 0
        _prob[k] = temp

    return _prob


# ### Laplace Smoothing

# In[100]:


# calculating probabilities with and without unknown word handling after 
uni_gram_prob = get_prob_after_smoothing_uni(unigram)
bi_gram_prob = get_prob_after_smoothing_bi(bi_gram, unigram)
uni_gram_prob_unk = get_prob_after_smoothing_uni(unigram_unk)
bi_gram_prob_unk = get_prob_after_smoothing_bi(bi_gram_unk, unigram_unk)


# In[101]:


uni_gram_prob['<unk>']


# In[102]:


uni_gram_prob_unk['<unk>']


# In[103]:


bi_gram_prob['booked']['<unk>']


# In[104]:


bi_gram_prob_unk['booked']['<unk>']


# Note: In add-k smoothing, probability of all words usually decreases slightly as a small value (k) is added to the count of each word for unseen words and make sure that no word has 0 probability.

# In[105]:


# example of probabilities before and after smoothing in unigram
test_key = "two"
print("Without smoothing :\t",unigram_probabilities[test_key])
print("With smoothing : \t",uni_gram_prob[test_key])


# In[106]:


# example of probabilities before and after smoothing in bigram
given = "i" 
find = "want"
print("Without smoothing :\t", bigram_probabilities[given][find])
print("With smoothing : \t", bi_gram_prob[given][find])


# ### Add-K smoothing

# In[108]:


# add-k with values = [0.005, .05, 2, 5]
uni_gram_k_005 = get_prob_after_smoothing_uni(unigram,0.005)
bi_gram_k_005 = get_prob_after_smoothing_bi(bi_gram_unk, unigram_unk,0.005)

uni_gram_k_05 = get_prob_after_smoothing_uni(unigram,0.05)
bi_gram_k_05 = get_prob_after_smoothing_bi(bi_gram_unk, unigram_unk,0.05)

uni_gram_k_2 = get_prob_after_smoothing_uni(unigram,2)
bi_gram_k_2 = get_prob_after_smoothing_bi(bi_gram, unigram_unk,2)

uni_gram_k_5 = get_prob_after_smoothing_uni(unigram,5)
bi_gram_k_5 = get_prob_after_smoothing_bi(bi_gram, unigram_unk,5)


# In[109]:


# example of probabilities before and after smoothing in unigram
test_key = "hello"
print("Without add-k smoothing :", unigram_probabilities[test_key])
print("With add-k smoothing : \t", uni_gram_k_005[test_key])


# In[110]:


# example of probabilities before and after smoothing in bigram
given = "i" 
find = "want"
print("Without add-k smoothing :\t",bigram_probabilities[given][find])
print("With add-k smoothing : \t",bi_gram_k_005[given][find])


# ## Part 3 - Calculating Perplexity 
# 
# ![image.png](attachment:image.png)

# In[111]:


test = open(pwd+"\\A1_DATASET\\val.txt","r", encoding='utf-8').read()
test = pre_process(test)
# test_unk = test.replace()
ts_tokens =  test.split(" ")
test_words = [x for x in ts_tokens if len(x)>0]


# In[112]:


# words in test set without unknown word handeling
test_words


# In[113]:


word_unk = list(unigram_unk.keys())
print(len(word_unk))
word_unk 


# In[114]:


test_words_unk = []
for word in test_words:
    if word not in word_unk:
        test_words_unk.append("<unk>")
    else: 
        test_words_unk.append(word)


# In[115]:


# test set words with handeled unknown words
test_words_unk


# In[119]:


len(test_words)


# In[121]:


len(list(unigram_probabilities_unk.keys()))


# In[116]:


# number of words in test test
N_test = 1/len(test_words)


# In[126]:


def unigram_perplexity(words, probabilites):
    perplexity = 0
    for i in words:
#         print("w: ",i," P",probabilites[i])
        try:
            perplexity += -math.log10(probabilites[i])
        except:
            perplexity += -math.log10(probabilites['<unk>'])
    perplexity *= N_test
    perplexity = pow(10,perplexity)
    return perplexity


# In[127]:


# perplexity for validation data using unsmoothed unigram training data
unigram_ppl_1 = unigram_perplexity(test_words_unk, unigram_probabilities_unk)
unigram_ppl_1


# In[128]:


def bigram_perplexity(sentences, probabilites):
    perplexity = 0
    for i in range(len(words)-1):
        try:
            perplexity += -math.log10(probabilites[words[i]][words[i+1]])
        except:
            perplexity += -math.log10(probabilites[words[i]][words[i+1]])
    perplexity *= N_test
    perplexity = pow(10,perplexity)
    return perplexity


# In[132]:


# perplexity for validation data using smoothed unigram training data
unigram_ppl_2 = unigram_perplexity(test_words_unk, uni_gram_prob_unk)
unigram_ppl_2


# In[133]:


# perplexity for validation data using smoothed unigram training data
unigram_ppl_3 = unigram_perplexity(test_words_unk, uni_gram_k_005)
unigram_ppl_3


# In[134]:


# perplexity for validation data using smoothed unigram training data
unigram_ppl_3 = unigram_perplexity(test_words_unk, uni_gram_k_05)
unigram_ppl_3


# In[135]:


# perplexity for validation data using smoothed unigram training data
unigram_ppl_3 = unigram_perplexity(test_words_unk, uni_gram_k_2)
unigram_ppl_3


# In[136]:


# perplexity for validation data using smoothed unigram training data
unigram_ppl_3 = unigram_perplexity(test_words_unk, uni_gram_k_5)
unigram_ppl_3


# In[138]:


# perplexity for validation data using unsmoothed bigram training data
bigram_ppl_1 = bigram_perplexity(test_words_unk,bigram_probabilities_unk)
bigram_ppl_1

# this will throw an error as the words have not occured in the training set


# In[137]:


# perplexity for validation data using unsmoothed bigram training data
bigram_ppl_2 = bigram_perplexity(test_words_unk,bi_gram_prob_unk)
bigram_ppl_2


# In[ ]:




