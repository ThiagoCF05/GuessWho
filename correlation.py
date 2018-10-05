__author__ = 'thiagocastroferreira'

import json
import nltk
import numpy as np
import operator
import re
import utils

import scikits.bootstrap as boot

def common_words_guesses(data):
    conditions = map(lambda x: x['condition'], data)
    voc, gvoc = {}, []
    for condition in set(conditions):
        voc[condition] = []
        f = filter(lambda guess: guess['condition'] == condition, data)

        for guess in f:
            for i, word in enumerate(guess['tokens']):
                if 'NN' in guess['pos_tag'][i] or 'NN' in guess['pos_tag'][i]:
                    voc[condition].append(word.lower())
                    gvoc.append(word.lower())

        # voc[condition] = nltk.FreqDist(voc[condition])
    tfidf = utils.tfidf(voc, 10)
    gvoc = nltk.FreqDist(gvoc)

    print 'Most common words in Guesses per condition:'
    for condition in tfidf:
        print 'Condition:', condition

        for word in tfidf[condition]:
            print word, gvoc[word[0]]
        print 20 * '-'

def common_words_faces(data):
    voc = []
    for face in data:
        for i, word in enumerate(face['tokens']):
            if 'NN' in face['pos_tag'][i] or 'JJ' in face['pos_tag'][i]:
                voc.append(word.lower())

    voc = nltk.FreqDist(voc)
    print 'Most common words in Faces:'
    v = sorted(voc.items(), key=operator.itemgetter(1), reverse=True)[:10]
    for word in v:
        print word
    print 20 * '-'

def common_words_faces_attractiveness(data):
    attractiveness = set(map(lambda face: face['responses']['attractive'], data))

    voc, gvoc = {}, []

    for attract in attractiveness:
        voc[attract] = []
        for face in filter(lambda face: face['responses']['attractive'] == attract, data):
            for i, word in enumerate(face['tokens']):
                if 'NN' in face['pos_tag'][i] or 'JJ' in face['pos_tag'][i]:
                    voc[attract].append(word.lower())
                    gvoc.append(word.lower())

    tfidf = utils.tfidf(voc, 10)
    gvoc = nltk.FreqDist(gvoc)

    print 'Most common words in Faces per attractiveness:'
    for attract in attractiveness:
        print attract
        for word in tfidf[attract]:
            print word, gvoc[word[0]]
        print 20 * '-'

def common_words_faces_typicality(data):
    typicality = set(map(lambda face: face['responses']['typical'], data))

    voc, gvoc = {}, []

    for typical in typicality:
        voc[typical] = []
        for face in filter(lambda face: face['responses']['typical'] == typical, data):
            for i, word in enumerate(face['tokens']):
                if 'NN' in face['pos_tag'][i] or 'JJ' in face['pos_tag'][i]:
                    voc[typical].append(word.lower())
                    gvoc.append(word.lower())

    tfidf = utils.tfidf(voc, 10)
    gvoc = nltk.FreqDist(gvoc)

    print 'Most common words in Faces per typicality:'
    for typical in typicality:
        print typical
        for word in tfidf[typical]:
            print word, gvoc[word[0]]
        print 20 * '-'

def attractiveness_per_race(data):
    races = set(map(lambda face: face['responses']['race'].lower(), data))
    voc = {}
    for race in races:
        voc[race] = []
        for face in filter(lambda face: face['responses']['race'].lower() == race, data):
            voc[race].append(face['responses']['attractive'])
        voc[race] = nltk.FreqDist(voc[race])

    print 'Attractiveness per race:'
    for race in races:
        v = sorted(voc[race].items(), key=operator.itemgetter(1), reverse=True)
        if len(v) > 2:
            print race
            for word in v:
                print word
            print 20 * '-'

# number of boards to finish the game (general and per condition)
def efficiency(data):
    games = set(map(lambda x: (x['uniqueID'], x['boardNumber']), data))

    boardCount = []
    boardCountCondition = {}
    boardCountGender = {
        'condition_male':{'participant_male':[], 'participant_female':[]},
        'condition_female':{'participant_male':[], 'participant_female':[]}
    }
    boardCountRace = {
        'condition_white': {},
        'condition_asian': {}
    }

    for game in games:
        boards = filter(lambda x: x['uniqueID'] == game[0] and x['boardNumber'] == game[1], data)
        boardCount.append(len(boards))

        if boards[0]['condition'] not in boardCountCondition:
            boardCountCondition[boards[0]['condition']] = []
        boardCountCondition[boards[0]['condition']].append(len(boards))

        if 'white' in boards[0]['condition'].lower():
            if boards[0]['demographics']['race'].lower() not in boardCountRace['condition_white']:
                boardCountRace['condition_white'][boards[0]['demographics']['race'].lower()] = []
            boardCountRace['condition_white'][boards[0]['demographics']['race'].lower()].append(len(boards))
        else:
            if boards[0]['demographics']['race'].lower() not in boardCountRace['condition_asian']:
                boardCountRace['condition_asian'][boards[0]['demographics']['race'].lower()] = []
            boardCountRace['condition_asian'][boards[0]['demographics']['race'].lower()].append(len(boards))

        if 'female' in boards[0]['demographics']['gender'].lower():
            if 'female' in boards[0]['condition'].lower():
                boardCountGender['condition_female']['participant_female'].append(len(boards))
            if 'male' in boards[0]['condition'].lower():
                boardCountGender['condition_male']['participant_female'].append(len(boards))
        else:
            if 'female' in boards[0]['condition'].lower():
                boardCountGender['condition_female']['participant_male'].append(len(boards))
            elif 'male' in boards[0]['condition'].lower():
                boardCountGender['condition_male']['participant_male'].append(len(boards))

    # print 'Average Number of Boards: ', np.mean(boardCount), np.std(boardCount)
    #
    # print '\nPer condition: '
    # for condition in boardCountCondition:
    #     print condition, np.mean(boardCountCondition[condition]), np.std(boardCountCondition[condition])
    #
    # print '\nPer gender: '
    # for condition in boardCountGender:
    #     for gender in boardCountGender[condition]:
    #         print condition, gender, np.mean(boardCountGender[condition][gender]), np.std(boardCountGender[condition][gender])
    #
    # print '\nPer race: '
    # for condition in boardCountRace:
    #     for race in boardCountRace[condition]:
    #         print condition, race, np.mean(boardCountRace[condition][race]), np.std(boardCountRace[condition][race])

    print 'Average Number of Boards: ', np.mean(boardCount), boot.ci(boardCount)

    print '\nPer condition: '
    for condition in boardCountCondition:
        print condition, np.mean(boardCountCondition[condition]), boot.ci(boardCountCondition[condition])

    print '\nPer gender: '
    for condition in boardCountGender:
        for gender in boardCountGender[condition]:
            print condition, gender, np.mean(boardCountGender[condition][gender]), boot.ci(boardCountGender[condition][gender])

    print '\nPer race: '
    for condition in boardCountRace:
        for race in boardCountRace[condition]:
            print condition, race, np.mean(boardCountRace[condition][race]), boot.ci(boardCountRace[condition][race])

def get_nps(data):
    def parse_np(index):
        np = ''
        closing = 0
        for elem in tree[index:]:
            if elem[0] == '(':
                closing += 1
            else:
                match = re.findall("\)", elem)

                np += elem.replace(')', '').strip() + ' '

                closing -= len(match)
                if closing <= 0:
                    break
        return np.replace('-LRB- ', '(').replace(' -RRB-', ')').replace('-LRB-', '(').replace('-RRB-', ')').strip().lower()

    nps = []
    nps_condition = {}
    for guess in data:
        tree = guess['parse'].split()
        for i, elem in enumerate(tree):
            if elem == '(NP':
                np = parse_np(i)
                nps.append(np)

                if guess['condition'] not in nps_condition:
                    nps_condition[guess['condition']] = []
                nps_condition[guess['condition']].append(np)

    # print 'Most frequent descriptions'
    # print 10 * '-'
    nps = nltk.FreqDist(nps)
    # v = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)[:30]
    # for np in v:
    #     print np[0], np[1]

    print 10 * '-'
    print 'Most distinctive descriptions per condition'
    nps_condition = utils.tfidf(nps_condition, 10)
    for condition in nps_condition:
        print 'Condition: ', condition
        print 10 * '-'
        for np in nps_condition[condition]:
            print np[0], np[1], nps[np[0]]
        print 10 * '-'

def get_nps_race(faces):
    def parse_np(index):
        np = ''
        closing = 0
        for elem in tree[index:]:
            if elem[0] == '(':
                closing += 1
            else:
                match = re.findall("\)", elem)

                np += elem.replace(')', '').strip() + ' '

                closing -= len(match)
                if closing <= 0:
                    break
        return np.replace('-LRB- ', '(').replace(' -RRB-', ')').replace('-LRB-', '(').replace('-RRB-', ')').strip().lower()

    nps = []
    nps_race = {}
    for face in faces:
        tree = face['parse'].split()
        for i, elem in enumerate(tree):
            if elem == '(NP':
                np = parse_np(i)
                nps.append(np)

                race = utils.process_race(face['responses']['race'].lower())

                if race not in nps_race:
                    nps_race[race] = []
                nps_race[race].append(np)

    print 10 * '-'
    print 'Most frequent descriptions per race'
    nps_race = utils.tfidf(nps_race, 10)
    nps = nltk.FreqDist(nps)
    for race in nps_race:
        print 'Race: ', race
        print 10 * '-'
        for np in nps_race[race]:
            print np[0], np[1], nps[np[0]]
        print 10 * '-'

def get_nps_attractiveness(faces):
    def parse_np(index):
        np = ''
        closing = 0
        for elem in tree[index:]:
            if elem[0] == '(':
                closing += 1
            else:
                match = re.findall("\)", elem)

                np += elem.replace(')', '').strip() + ' '

                closing -= len(match)
                if closing <= 0:
                    break
        return np.replace('-LRB- ', '(').replace(' -RRB-', ')').replace('-LRB-', '(').replace('-RRB-', ')').strip().lower()

    nps = []
    nps_attractiveness = {}
    for face in faces:
        tree = face['parse'].split()
        for i, elem in enumerate(tree):
            if elem == '(NP':
                np = parse_np(i)
                nps.append(np)

                attract = face['responses']['attractive'].lower()

                if attract not in nps_attractiveness:
                    nps_attractiveness[attract] = []
                nps_attractiveness[attract].append(np)

    print 10 * '-'
    print 'Most frequent descriptions per attractiveness'
    nps_attractiveness = utils.tfidf(nps_attractiveness, 10)
    nps = nltk.FreqDist(nps)
    for attract in nps_attractiveness:
        print 'Race: ', attract
        print 10 * '-'
        for np in nps_attractiveness[attract]:
            print np[0], np[1], nps[np[0]]
        print 10 * '-'

def question_size(data):
    nps_question = {}
    nps_boardQuestion = {}
    nps_conditionQuestion = {}

    for guess in data:
        if guess['questionNumber'] not in nps_question:
            nps_question[guess['questionNumber']] = []
        nps_question[guess['questionNumber']].append(len(guess['tokens']))

        if (guess['boardNumber'], guess['questionNumber']) not in nps_boardQuestion:
            nps_boardQuestion[(guess['boardNumber'], guess['questionNumber'])] = []
        nps_boardQuestion[(guess['boardNumber'], guess['questionNumber'])].append(len(guess['tokens']))

        if guess['condition'] not in nps_conditionQuestion:
            nps_conditionQuestion[guess['condition']] = {}

        if guess['questionNumber'] not in nps_conditionQuestion[guess['condition']]:
            nps_conditionQuestion[guess['condition']][guess['questionNumber']] = []
        nps_conditionQuestion[guess['condition']][guess['questionNumber']].append(len(guess['tokens']))

    print 10 * '-'
    print 'Size of question during a board'
    for questionNumber in sorted(nps_question.keys()):
        print questionNumber, np.mean(nps_question[questionNumber]), np.std(nps_question[questionNumber])

    print 10 * '-'
    print 'Size of question during a game'
    for key in sorted(nps_boardQuestion.keys(), key=lambda x: (x[1], x[0])):
        print key, np.mean(nps_boardQuestion[key]), np.std(nps_boardQuestion[key])

    print 10 * '-'
    print 'Size of question during a board from a condition'
    for condition in nps_conditionQuestion:
        print condition
        print 10 * '-'
        for questionNumber in sorted(nps_conditionQuestion[condition].keys()):
            print questionNumber, np.mean(nps_conditionQuestion[condition][questionNumber]), np.std(nps_conditionQuestion[condition][questionNumber])
        print 10 * '-'

if __name__ == '__main__':
    faces = json.load(open('../Guess-who-dataset/face-rating-data.json'))
    guesses = json.load(open('../Guess-who-dataset/guess-who-data.json'))

    # common_words_guesses(guesses)
    # common_words_faces(faces)
    # common_words_faces_attractiveness(faces)
    # common_words_faces_typicality(faces)
    # attractiveness_per_race(faces)

    # efficiency(guesses)
    # get_nps(guesses)

    question_size(guesses)

    # get_nps_race(faces)
    # get_nps_attractiveness(faces)