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


bigDic = {}
docFreq = []
for i, doc in enumerate(preprocessedDocs):
    docFreq.append(dict())

    for j, term in enumerate(doc):
        
        if term in bigDic:
            bigDic[term][0] += 1
        else:
            bigDic[term] = []
            bigDic[term].append(1)
            bigDic[term].append(dict())
        
        if term in docFreq[i]:
            docFreq[i][term] += 1
        else:
            docFreq[i][term] = 1
            bigDic[term][1][i] = []
        bigDic[term][1][i].append(j) #positions

bigDicSorted = OrderedDict(sorted(bigDic.items()))

####################################################################### 
##############           part one end             #####################
#######################################################################


def sortBasedOnValue(Dic):
    keys = list(Dic.keys())
    values = list(Dic.values())
    sorted_value_index = np.argsort(values)
    return {keys[i]: values[i] for i in sorted_value_index}


def intersection(postings1: list, postings2: list):
    return list(set(postings1) & set(postings2))

def and_not(postings1: list, postings2: list):
    return [x for x in postings1 if x not in postings2]

def union(postings1: list, postings2: list):
    return list(set(postings1) | set(postings2))


# query = input("your query: ")
query = 'مایکل ! جردن'
# query = 'باشگاه¬های فوتسال آسیا'
# query = '"سهمیه المپیک"'
normalized_query = normalizer.normalize(query)
tokenized_query = tokenizer.tokenize_words(normalized_query)
tokenized_query = list(filter(not_remove, tokenized_query))
tokenized_query = list(map(stemmer.convert_to_stem, tokenized_query)) #swap from phase 1.1
 
print("tokenized query: {}".format(tokenized_query))

postings_tmp = []
not_flg = False
phrase_flg = False
phrase = []
query_words = []
not_found = False
for i, tkn in enumerate(tokenized_query):
    if tkn == '!':
        not_flg = True
    elif not_flg:
        postings_tmp = and_not(postings_tmp, bigDicSorted[tkn][1].keys())
        not_flg = False
    elif tkn == '"':
        phrase_flg = not phrase_flg
    elif phrase_flg:
        phrase.append(tkn)
    else:
        if tkn not in bigDic:
            not_found = True
        else:
            query_words.append(tkn)
            if len(postings_tmp) == 0:
                postings_tmp = union(postings_tmp, bigDicSorted[tkn][1].keys())
            else:
                postings_tmp = intersection(postings_tmp, bigDicSorted[tkn][1].keys())

if not_found:
    postings_tmp.clear()
    
phrase_query_result = []
phrase_res_dict = {} #keys: doc id, val : freq
if len(phrase) != 0:
    postings_lists = [bigDicSorted[term][1] for term in phrase]
    doc_ids = set(postings_lists[0].keys())
    for postings in postings_lists[1:]:
        doc_ids = doc_ids.intersection(postings.keys())
    matches = []
    for doc_id in doc_ids:
        positions = []
        for term, postings in zip(phrase, postings_lists):
            positions.append(postings[doc_id])
        for pos in positions[0]:
            if all(pos+i in positions[i] for i in range(1, len(positions))):
                matches.append((doc_id, pos))
    phrase_res_dict = {} #keys: doc id, val : freq
    for match in matches:
        doc_id, pos = match
        if doc_id in phrase_res_dict:
            phrase_res_dict[doc_id] += 1
        else:
            phrase_res_dict[doc_id] = 1

    # sorted_phrase_dict = sortBasedOnValue(phrase_res_dict)
    if len(postings_tmp) == 0:
        postings_tmp = phrase_res_dict.keys()
    else:
        postings_tmp = intersection(postings_tmp, phrase_res_dict.keys())
    


unrelated_docs = []
if len(postings_tmp) < 5:
        for w in query_words:
            unrelated_docs = union(postings_tmp, bigDicSorted[w][1].keys())
    

def sortResults(postings):
    tempDict = {}
    for docId in postings:
        sum = 0
        if docId in phrase_res_dict:
            sum+=phrase_res_dict[docId]
        for w in query_words:
            sum += docFreq[docId][w]
        tempDict[docId] = sum
    tempDict = sortBasedOnValue(tempDict)
    newList = list(tempDict.keys())
    newList.reverse()
    return newList, tempDict

res, res_dict = sortResults(postings_tmp)
for i, k in enumerate(res):
    if i > 4:
        break
    print('{}.doc: {}\tfreqs: {}'.format(i+1, k, res_dict[k]))
    print('title: {}\nurl: {}'.format(data.iloc[k]["title"], data.iloc[k]["url"]))

unrelated_docs_new, unrelated_docs_dict = sortResults(unrelated_docs)
for i, doc in enumerate(unrelated_docs_new):
    print("*******************"*5)
    if i > 4:
        break
    print('{}.doc: {}\tfreqs: {}\t(this doc is not completely related)'.format(i+1, doc, unrelated_docs_dict[doc]))
    print('title: {}\nurl: {}'.format(data.iloc[doc]["title"], data.iloc[doc]["url"]))
