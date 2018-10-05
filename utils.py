import nltk
import numpy as np
import operator

def process_race(aux):
    if 'hispanic' in aux or 'mexi' in aux or 'latin' in aux or 'mestizo' in aux:
        race = 'latin'
    elif 'asia' in aux or 'chin' in aux or 'asai' in aux or 'aisi' in aux or 'japan' in aux or 'korea' in aux:
        race = 'asian'
    elif 'india' in aux or 'india' in aux:
        race = 'indian'
    elif 'middle' in aux or 'arab' in aux or 'muslim' in aux:
        race = 'middle_eastern'
    elif 'black' in aux or 'africa' in aux:
        race = 'black'
    elif 'mix' in aux or 'biracial' in aux:
        race = 'mixed'
    elif 'white' in aux or 'cauca' in aux or 'europ' in aux:
        race = 'white'
    else:
        race = 'other'
    return race

def frequency(voc):
    freq = {}
    for condition in voc:
        freq[condition] = nltk.FreqDist(voc[condition])
    return freq

def tfidf(voc, beam=30):
    freq = {}
    for condition in voc:
        freq[condition] = nltk.FreqDist(voc[condition])

    _tfidf = {}
    for condition in freq:
        dem = float(sum(freq[condition].values()))
        _tfidf[condition] = dict(map(lambda x: (x[0], (x[1] / dem)), freq[condition].items()))

        for word in _tfidf[condition]:
            idf = np.log(float(len(freq.keys())) / len(filter(lambda c: word in freq[c], freq.keys())))
            _tfidf[condition][word] = _tfidf[condition][word] * idf
        _tfidf[condition] = sorted(_tfidf[condition].items(), key=operator.itemgetter(1), reverse=True)[:beam]
    return _tfidf