__author__ = 'thiagocastroferreira'

import json

from stanford_corenlp_pywrapper import CoreNLP

if __name__ == '__main__':
    proc = CoreNLP('parse')

    faces = json.load(open('../Guess-who-dataset/face-rating-data.json'))
    guesses = json.load(open('../Guess-who-dataset/guess-who-data.json'))

    for guess in guesses:
        if guess['actionType'] == 'questionAsked':
            out = proc.parse_doc(guess['question'].replace('[comma]', ','))
            if len(out['sentences']) > 0:
                guess['tokens'] = out['sentences'][0]['tokens']
                guess['pos_tag'] = out['sentences'][0]['pos']
                guess['parse'] = out['sentences'][0]['parse']
            else:
                guess['tokens'] = []
                guess['pos_tag'] = []
                guess['parse'] = ''
        else:
            guess['tokens'] = []
            guess['pos_tag'] = []
            guess['parse'] = ''

    for face in faces:
        out = proc.parse_doc(face['responses']['description'].replace('[comma]', ','))
        face['tokens'] = out['sentences'][0]['tokens']
        face['pos_tag'] = out['sentences'][0]['pos']
        face['parse'] = out['sentences'][0]['parse']

    json.dump(guesses, open('../Guess-who-dataset/guess-who-data.json', 'w'), indent=4, separators=(',', ': '))
    json.dump(faces, open('../Guess-who-dataset/face-rating-data.json', 'w'), indent=4, separators=(',', ': '))