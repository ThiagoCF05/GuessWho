__author__ = 'thiagocastroferreira'

import json

from stanfordcorenlp import StanfordCoreNLP

STANFORD_PATH=r'/home/tcastrof/workspace/stanford/stanford-corenlp-full-2018-02-27'

if __name__ == '__main__':
    props={'annotators': 'tokenize,ssplit,pos,lemma,parse','pipelineLanguage':'en','outputFormat':'json'}
    corenlp = StanfordCoreNLP(STANFORD_PATH)

    faces = json.load(open('dataset/face-rating-data-2.json'))
    guesses = json.load(open('dataset/guess-who-data-2.json'))

    for guess in guesses:
        if guess['actionType'] == 'questionAsked':
            out = corenlp.annotate(guess['question'].replace('[comma]', ','), properties=props)
            parsed = json.loads(out)
            guess['nlp'] = {}

            tokens, pos_tag, tree = [], [], ''
            for sentence in out['sentences']:
                tokens.extend([w['originalText'] for w in sentence['tokens']])
                pos_tag.extend([w['pos'] for w in sentence['tokens']])
                tree += ' ' + sentence['parse']

            guess['nlp']['tokens'] = tokens
            guess['nlp']['pos_tag'] = pos_tag
            guess['nlp']['parse'] = tree.strip()
        else:
            guess['nlp']['tokens'] = []
            guess['nlp']['pos_tag'] = []
            guess['nlp']['parse'] = ''

    for face in faces:
        out = corenlp.annotate(face['responses']['description'].replace('[comma]', ','), properties=props)
        parsed = json.loads(out)

        tokens, pos_tag, tree = [], [], ''
        for sentence in out['sentences']:
            tokens.extend([w['originalText'] for w in sentence['tokens']])
            pos_tag.extend([w['pos'] for w in sentence['tokens']])
            tree += ' ' + sentence['parse']

        face['nlp'] = {}
        face['nlp']['tokens'] = tokens
        face['nlp']['pos_tag'] = pos_tag
        face['nlp']['parse'] = tree.strip()

    corenlp.close()

    json.dump(guesses, open('dataset/guess-who-data-2.json', 'w'), indent=4, separators=(',', ': '))
    json.dump(faces, open('dataset/face-rating-data-2.json', 'w'), indent=4, separators=(',', ': '))