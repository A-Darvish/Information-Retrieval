# import pickle
import math
import numpy as np
import hazm
import pandas # library for working with large JSON file
from parsivar import Normalizer, Tokenizer, FindStems
from collections import defaultdict, OrderedDict

path = "./IR_data_news_5k.json"



data = pandas.read_json(path)
data = data.T

stopWords = hazm.stopwords_list()
normalizer = Normalizer()
stemmer = FindStems()
tokenizer = Tokenizer()
punctuations = ['،', '.', '»', '«', '؟', '(', ')', '/'] #edited from phase 1.1


def not_remove(word):
    return not ((word in stopWords) or (word in punctuations))



preprocessedDocs = []
for doc in data.itertuples():
    normalized_doc = normalizer.normalize(doc.content)
    tokenized_doc = tokenizer.tokenize_words(normalized_doc)
    entries = list(filter(not_remove, tokenized_doc))
    stemmed = list(map(stemmer.convert_to_stem, entries)) #swap from phase 1.1
    
    preprocessedDocs.append(stemmed)

posDics = []

for i, doc in enumerate(preprocessedDocs):
    posDics.append(defaultdict(list))
    for j, token in enumerate(doc):
        posDics[i][token].append(j)



bigDic = {}

docFreq = [] 
# is a list to store frequency of each term in each doc,
# for ex: docFreq[0] is a dictionary that have the terms of doc id 0 and its frequency in doc 0.

for i, doc in enumerate(preprocessedDocs):
    docFreq.append(dict())

    for j, term in enumerate(doc):
        
        if term in bigDic:
            bigDic[term][0] += 1
        else:
            bigDic[term] = []
            bigDic[term].append(1) # total freq of term in whole collection
            bigDic[term].append(dict()) # postings lists
        
        if term in docFreq[i]:
            docFreq[i][term] += 1
        else:
            docFreq[i][term] = 1
            bigDic[term][1][i] = []
            bigDic[term][1][i].append(list())
        bigDic[term][1][i][0].append(j) #positions
    

N = len(preprocessedDocs)    

for i, doc in enumerate(preprocessedDocs):
    for j, term in enumerate(doc):
        tf = 1 + math.log10(docFreq[i][term])
        df = len(bigDic[term][1])
        idf = math.log10(N/df)
        bigDic[term][1][i].append(tf*idf)


def sortBasedOnValue(Dic):
    keys = list(Dic.keys())
    values = list(Dic.values())
    sorted_value_index = np.argsort(values)
    return {keys[i]: values[i] for i in sorted_value_index}

champion_list_param = 50
champion_lists= {}

for term in bigDic:
    old_tmpDic = bigDic[term][1]
    keys = list(old_tmpDic.keys())
    tmpDic = {keys[i]: old_tmpDic[k][1] for i, k in enumerate(old_tmpDic)}
    champion_lists[term] = list(sortBasedOnValue(tmpDic))[-champion_list_param:]
    champion_lists[term].reverse()   # for sorting from highest to lowest.

# print(champion_lists['پنجعلی'])

bigDicSorted = OrderedDict(sorted(bigDic.items()))

# pickleFile_write = open('filePickle', 'ab')
# pickle.dump(bigDicSorted, pickleFile_write)					
# pickleFile_write.close()


def intersection(postings1: list, postings2: list):
    return list(set(postings1) & set(postings2))

def and_not(postings1: list, postings2: list):
    return [x for x in postings1 if x not in postings2]

def union(postings1: list, postings2: list):
    return list(set(postings1) | set(postings2))

# query = input("your query: ")
query = 'مایکل جردن'
# query = 'تنویر افکار عمومی'
# query = 'باشگاه¬های فوتسال آسیا'
normalized_query = normalizer.normalize(query)
tokenized_query = tokenizer.tokenize_words(normalized_query)
tokenized_query = list(filter(not_remove, tokenized_query))
tokenized_query = list(map(stemmer.convert_to_stem, tokenized_query)) #swap from phase 1.1

jaccard = True
with_champion = False

print("tokenized query: {}\tchampion = {}".format(tokenized_query, with_champion))

# pickleFile_read = open('filePickle', 'rb')	
# bigDicSorted = pickle.load(pickleFile_read)


if jaccard:
    my_query = set(tokenized_query)

    jaccard_score = {}
    def jaccard_similarity(set1, set2):
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        similarity = len(intersection) / len(union)
        return similarity


    
    for term in my_query:
        if with_champion :
            postings = champion_lists[term]
        else:
            postings = bigDicSorted[term][1]
        for doc in postings:
            if doc in jaccard_score:
                continue
            doc_set = set(preprocessedDocs[doc])
            jaccard_score[doc] = jaccard_similarity(my_query, doc_set)


    jaccard_scoreSorted = sortBasedOnValue(jaccard_score)
    jaccard_newList = list(jaccard_scoreSorted.keys())
    jaccard_newList.reverse()


    for i, k in enumerate(jaccard_newList):
        if i > 4:
            break
        print('{}.doc: {}\tjaccard score: {}'.format(i+1, k, jaccard_score[k]))
        print('title: {}\nurl: {}'.format(data.iloc[k]["title"], data.iloc[k]["url"]))

print('-------------------------------------------------------------------------------')
qFreq = {}
for term in tokenized_query:
    if term not in qFreq:
        qFreq[term] = 1
    else:
        qFreq[term] += 1

score = {}
length = {}
checkedTerm = set()
for term in tokenized_query:
    if term in checkedTerm:
        continue
    checkedTerm.add(term)
    if with_champion :
        postings = champion_lists[term]
    else:
        postings = bigDicSorted[term][1]
    tf = 1 + math.log10(qFreq[term])
    df = len(bigDicSorted[term][1])
    idf = math.log10(N/df)
    w = tf * idf

    for doc in postings:
        if doc in score:
            score[doc] += w * bigDicSorted[term][1][doc][1]
        else:
            score[doc] = w * bigDicSorted[term][1][doc][1]
        if doc not in length:
            length[doc] = 0
            
for doc in length :
    doc_set = set(preprocessedDocs[doc])
    for term in doc_set:
        length[doc] += bigDicSorted[term][1][doc][1] * bigDicSorted[term][1][doc][1]
for doc in score:
    score[doc] /= math.sqrt(length[doc])
scoreSorted = sortBasedOnValue(score)
newList = list(scoreSorted.keys())
newList.reverse()

for i, k in enumerate(newList):
    if i > 4:
        break
    print('{}.doc: {}\tcosine score: {}'.format(i+1, k, score[k]))
    print('title: {}\nurl: {}'.format(data.iloc[k]["title"], data.iloc[k]["url"]))

# pickleFile_read.close()
