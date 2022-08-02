#!/usr/bin/python3

import re
import scipy as sp
import numpy as np
import scipy.sparse
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

def tokenize(text, space = True):
    tokens = []
    for token in re.split("([0-9a-zA-Z'-]+)", text):
        if not space:
            token = re.sub("[ ]+", "", token)
        if not token:
            continue
        if re.search("[0-9a-zA-Z'-]", token):                    
            tokens.append(token)
        else: 
            tokens.extend(token)
    return tokens

def sentokenize(text, space = True):
    sentences = []
    for sentence in re.split("(\s+(?<=[.?!,;:\n][^a-zA-Z0-9])\s*)", text):
        if (len(sentence)==1 and not re.search("[0-9a-zA-Z'-]", sentence[0])):
            if len(sentences):
                sentences[-1] = sentences[-1] + [sentence]  
            else:
                sentences.append([sentence])
        elif not re.search("[0-9a-zA-Z'-]", sentence):
            tokens = tokenize(sentence, space = space)
            if len(sentences):
                sentences[-1] = sentences[-1] + tokens  
            else:
                sentences.append(tokens)
        else:
            sentences.append(tokenize(sentence, space = space))
    return sentences

def make_TDM(documents, do_tfidf = True, space = True, normalize = True):
    document_frequency = Counter()
    for j, document in enumerate(documents):
        frequency = Counter([t for s in sentokenize(document.lower(), space = space) 
                         for t in s])
        document_frequency += Counter(frequency.keys())
    type_index = {t:i for i, t in enumerate(sorted(list(document_frequency.keys())))}
    document_frequency = np.array(list(document_frequency.values()))
    # performs the counting again, and stores with standardized indexing`
    counts, row_ind, col_ind = map(np.array, zip(*[(count, type_index[t],j) 
                                                   for j, document in enumerate(documents) 
                                                   for t, count in Counter(tokenize(document.lower(), space = space)).items()]))
    # constructs a sparse TDM from the indexed counts
    TDM = sp.sparse.csr_matrix((counts, (row_ind, col_ind)),
                             shape = (len(document_frequency),len(documents)))
    if normalize:
        # normalize frequency to be probabilistic
        TDM = TDM.multiply(1/TDM.sum(axis = 0))
    # apply tf-idf
    if do_tfidf:
        num_docs = TDM.shape[1]
        IDF = -np.log2(document_frequency/num_docs)
        TDM = (TDM.T.multiply(IDF)).T
    return(TDM, type_index)

def get_context(i, sentence, k = 20, gamma = 0):
    context = np.array(sentence)
    weights = np.abs(np.array(range(len(sentence))) - i)
    mask = (weights != 0) & (weights <= k) if k else (weights != 0)
    context = context[mask]
    weights = 1/(weights[mask]**gamma) if gamma else weights[mask]*gamma + 1.
    return context, weights

def make_CoM(documents, k = 20, gamma = 0, space = True, do_tficf = True, normalize = True):
    document_frequency = Counter()
    for j, document in enumerate(documents):
        sentences = sentokenize(document.lower(), space = space)
        documents[j] = sentences
        frequency = Counter([t for s in documents[j] for t in s])
        document_frequency += Counter(frequency.keys())
    type_index = {t:i for i, t in enumerate(sorted(list(document_frequency.keys())))}

    co_counts = Counter()  
    for document in documents:
        for sentence in document:
            for i, ti in enumerate(sentence):
                context, weights = get_context(i, sentence, k = k, gamma = gamma)        
                for j, tj in enumerate(context):
                    co_counts[(type_index[ti], type_index[tj])] += weights[j]

    type_ijs, counts = zip(*co_counts.items())
    row_ind, col_ind = zip(*type_ijs)

    # constructs a sparse CoM from the indexed counts
    CoM = sp.sparse.csr_matrix((counts, (row_ind, col_ind)),
                              shape = (len(type_index),len(type_index)))
    if normalize:
        # normalize frequency to be probabilistic
        CoM = CoM.multiply(1/CoM.sum(axis = 0))
    
    # apply tf-icf
    if do_tficf:
        context_frequency = np.count_nonzero(CoM.toarray(), axis=1)
        num_cons = CoM.shape[1]
        ICF = -np.log2(context_frequency/num_cons)
        CoM = (CoM.T.multiply(ICF)).T
        
    return CoM, type_index

def svdsub(X, d=50, state = 691):
    return TruncatedSVD(n_components=d, random_state = state).fit_transform(X)

def eval_tweet_prediction(TDM, newstweet, state = 0):
    y = np.array([int(bool(x['tweets'])) for x in newstweet])
    x = TDM.T
    x_train, x_test_holdout, y_train, y_test_holdout = train_test_split(x, y, test_size=0.33, random_state=state)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.67, random_state=state)

    classifier = LogisticRegression(solver='liblinear', random_state = state)
    classifier.fit(x_train, y_train)
    prediction_probabilities = classifier.predict_proba(x_test)

    max_F1 = 0
    max_ix = 0
    max_threshold = 0
    for ix, threshold in enumerate(np.array(range(1,1001))/1000):
        threshold_predictions = np.array([1 if prediction[1] > threshold else 0 
                                          for prediction in prediction_probabilities])
        TP = ((threshold_predictions == 1) & (y_test == 1)).astype(int).sum()
        FP = ((threshold_predictions == 1) & (y_test == 0)).astype(int).sum()
        TN = ((threshold_predictions == 0) & (y_test == 0)).astype(int).sum()
        FN = ((threshold_predictions == 0) & (y_test == 1)).astype(int).sum()
        P = TP/(TP + FP) if TP + FP else 0
        R = TP/(TP + FN) if TP + FN else 0
        F1 = ((2*P*R)/(P+R)) if P+R else 0
        if F1 > max_F1 and R != 1.0:
            max_F1 = F1
            max_ix = ix
            max_threshold = threshold

    holdout_predictions = classifier.predict_proba(x_test_holdout)
    threshold_predictions = np.array([1 if prediction[1] > max_threshold else 0 
                                    for prediction in holdout_predictions])
    TP = ((threshold_predictions == 1) & (y_test_holdout == 1)).astype(int).sum()
    FP = ((threshold_predictions == 1) & (y_test_holdout == 0)).astype(int).sum()
    TN = ((threshold_predictions == 0) & (y_test_holdout == 0)).astype(int).sum()
    FN = ((threshold_predictions == 0) & (y_test_holdout == 1)).astype(int).sum()
    P = TP/(TP + FP) if TP + FP else 0
    R = TP/(TP + FN) if TP + FN else 0
    F1 = ((2*P*R)/(P+R)) if P+R else 0

    return Counter({"P": P, "R": R, "F1": F1}), classifier, max_threshold

def cbow(TDM, CoM):
    return TDM.T.dot(CoM).T