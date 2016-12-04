
# coding: utf-8

# In[1]:



"""
CS579: Assignment 2
In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.
You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.
The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.
Complete the 14 methods below, indicated by TODO.
As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.
    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.
    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.
    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], 
          dtype='<U5')
    """
    test=re.sub(r'[\n\r\x85]', ' ', doc.lower())
    if(keep_internal_punct):
        x=re.findall(r"[\w\']+", test.lower())
    else:
        x=re.findall(r"[\w]+", test.lower())
    return np.array(x)


def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.
    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    C =Counter(tokens)
    for i in set(tokens):
        feats['token='+i]+=C[i]


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.
    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)
    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.
    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    i=0
    j=0
    window = []
    def add(A,B,New):
        if(New):
            feats['token_pair='+A+'__'+B]=1
        else:
            feats['token_pair='+A+'__'+B] +=1
    for l in range(0, len(tokens)-2):
        window = tokens[l:l+k]
        for i in range(len(window)-1):
            for j in range(i+1, len(window)):
                if 'token_pair='+window[i]+'__'+window[j] in feats:
                    if(window[i]!=window[j]):
                        if(window[i]<window[j]):
                            add(window[i],window[j],False)
                        else:
                            add(window[j],window[i],False)
                elif 'token_pair='+window[j]+'__'+window[i] in feats:
                    if(window[i]!=window[j]):
                        if(window[i]<window[j]):
                            add(window[i],window[j],False)
                        else:
                            add(window[j],window[i],False)
                else:
                    if(window[i]!=window[j]):
                        if(window[i]<window[j]):
                            add(window[i],window[j],True)
                        else:
                            add(window[j],window[i],True)


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.
    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    feats['neg_words']=0
    feats['pos_words']=0
    for i in tokens:
        if(i.lower() in neg_words):
            feats['neg_words']+=1
        elif(i.lower() in pos_words):
            feats['pos_words']+=1


def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.
    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.
    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    feats = defaultdict(lambda: 0)
    for i in feature_fns:
        i(tokens, feats)
    return sorted(feats.items())

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.
    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),
    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    feats = defaultdict(lambda: 0)
    docs = defaultdict(lambda: 0)
    voca=defaultdict(lambda: 0)
    frequency=defaultdict(lambda: 0)
    copier=defaultdict(lambda: 0)
    data=[]
    row=[]
    col=[]
    C=0
    #Vocab for min_freq. For every Doc update new Dictionary indicating Token frequency in overall list of Docs
    for i in tokens_list:    
        for j in feature_fns:
            j(i,feats)
            for k,v in feats.items():
                if(copier[k]!=feats[k]):
                    frequency[k]+=1
            copier=feats.copy()
    for k,v in frequency.copy().items():
        if(v<min_freq):
            del frequency[k]
    #removing keys which dont have minimum frequency
    sender=frequency.copy()
    #use as Model for CSR Labelling
    featslist=  dict.fromkeys(sender.keys(),0)
    feats = defaultdict(lambda: 0)
    for counti,i in enumerate(tokens_list):
        for j in feature_fns:
            j(i,feats)
        for k,v in feats.items():
            if(k in featslist.keys()):
                featslist[k]=v
        for counta,a in enumerate(sorted(featslist.items())):
            if(a[1]!=0):
                data.append(a[1])
                row.append(counti)
                col.append(counta)
        #Re-Initialize Model Dict to zero for new run
        featslist= dict.fromkeys(sender.keys(),0)
        feats=defaultdict(lambda: 0)
    C=0
    if(vocab==None):
        for k,v in sorted(sender.items()):
            voca[k]=C
            C+=1
    else:
        voca=vocab.copy()
    #print((data, (row, col)))
    return(csr_matrix((data, (row, col)),shape=(len(tokens_list),len(voca)), dtype=np.int64),voca)

def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).
    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.
    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    cv = KFold(len(labels),k)
    acc = []
    n=0
    for train_idx, test_idx in cv:
        clf.fit(X[train_idx], labels[train_idx])
        predicted = clf.predict(X[test_idx])
        accuracy = len(np.where(labels[test_idx]==predicted)[0]) / len(labels[test_idx])
        acc.append(accuracy)
    avg = np.mean(acc)
    return avg


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.
    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.
    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).
    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])
    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.
      This list should be SORTED in descending order of accuracy.
      This function will take a bit longer to run (~20s for me).
    """
    l=[]
    d=defaultdict()
    dd=defaultdict()
    c=0
    combs=[]
    for i in range(1, len(feature_fns)+1):
        els = [list(x) for x in combinations(feature_fns, i)]
        combs.extend(els)
    
    for punct in punct_vals:
        tokens_list = [tokenize(d,punct) for d in docs]
        for feat in combs:
            for freq in min_freqs:
                X, vocab=vectorize(tokens_list, feat, freq)
                accur=cross_validation_accuracy(LogisticRegression(), X, labels,5)
                d['features']=feat
                d['punct']=punct
                d['accuracy']=accur
                d['min_freq']=freq
                l.append(sorted(d.items()))
    return(sorted(l,key=lambda x:-x[0][1])) 


def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    aa=[]
    for counta,a in enumerate(results):
        aa.append(a[0][1])
    aa.reverse()

    plt.xlabel('k')
    plt.ylabel('accuracy')    
    plt.plot(np.arange(42)+1,aa)
    plt.savefig("accuracies.png",dpi=1200)


def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.
    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    aa=defaultdict(list)
    aaa=[]
    for a in results:
        b=str("feature=")
        for i in tokenize(str(a[1][1]), True):
            if(re.search(r'\_', i)):
                b+=i+" "
        aa["punct="+str(a[2][1])].append(a[0][1])
        aa[b].append(a[0][1])
        aa["min_freq="+str(a[3][1])].append(a[0][1])
    #print(aa)
    for i in sorted(aa.items(),key=lambda x:-np.mean(x[1])):
        aaa.append((float(np.mean(i[1])),i[0]))
    #print(aaa)
    return(aaa)


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)
    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    tokens_list = [tokenize(d,best_result[3][1]) for d in docs]
    XX, vocabb=vectorize(tokens_list, best_result[1][1], best_result[2][1])
    #print(len(vocabb))
    #print(XX.shape[0],XX.shape[1])
    model=LogisticRegression()
    model.fit(XX,labels)
    return(model,vocabb)


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.
    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    aa=[]
    co=list(clf.coef_)[0]
    ##print(co)
    if(len(co)==len(vocab)):
        for k,v in vocab.items():
            aa.append((k,co[v]))
    #print(sorted(aa,key=lambda x:x[1])[:5])
    if(label==1):
        return((t,v) for t,v in sorted(aa,key=lambda x:-x[1])[:n])
    if(label==0):
        return((t,-v) for t,v in sorted(aa,key=lambda x:x[1])[:n])


def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.
    Note: use read_data function defined above to read the
    test data.
    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    docs, labels = read_data(os.path.join('data', 'test'))
    tokens_list = [tokenize(d,best_result[3][1]) for d in docs]
    XX, vocabb=vectorize(tokens_list, best_result[1][1], best_result[2][1],vocab)
    return(docs,labels,XX)


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.
    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.
    Returns:
      Nothing; see Log.txt for example printed output.
    """
    ###TODO
    pass
    aa=[]
    a=clf.predict_proba(X_test)
    b=clf.predict(X_test)
    for c,i in enumerate(a):
        if(test_labels[c]!=b[c]):
            #print(test_labels[c],b[c],i)
            if(test_labels[c]>b[c]):
                aa.append((i[0],test_labels[c],b[c],c))
            else:
                aa.append((i[1],test_labels[c],b[c],c))
    for i in sorted(aa,key=lambda x:-x[0])[:n]:
        print("truth=%s predicted=%s proba= %.5f" %(i[1],i[2],i[0]))
        print(test_docs[i[3]])


def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()

