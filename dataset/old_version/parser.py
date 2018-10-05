__author__ = 'thiagocastroferreira'
import json
import re

def guess_who_parser():
    f = open('guess-who-data.csv')
    data = f.read().split('\r\n')
    f.close()

    result = []

    for i, line in enumerate(data):
        _json, date, number = line.split('\t')
        _json = re.sub('\"+', '\"', _json[1:-1])
        _json = _json.replace('\"eliminatedFaces\":\",', '\"eliminatedFaces\":\"\",').replace('\"question":\",',
                                                                                              '\"question":\"\",')

        print i, _json
        print '\n'
        _json = json.loads(_json)

        _json['allFaces'] = _json['allFaces'].split('__')
        _json['eliminatedFaces'] = _json['eliminatedFaces'].split('__')
        # demographics
        _demographics = _json['demographics'].split('___')
        demographics = {}
        for elem in _demographics:
            key, value = elem.split('=')
            demographics[key] = value
        _json['demographics'] = demographics

        result.append(_json)

    json.dump(result, open('guess-who-data.json', 'w'), indent=4, separators=(',', ': '))

def face_rating():
    f = open('face-rating-data.csv')
    data = f.read().split('\r\n')
    f.close()

    result = []
    for i, line in enumerate(data):
        _json, date, number = line.split('\t')
        _json = re.sub('\"+', '\"', _json[1:-1])

        print i, _json
        print '\n'
        _json = json.loads(_json)

        # responses
        _responses = _json['responses'].split('___')
        responses = {}
        for elem in _responses:
            key, value = elem.split('=')
            responses[key] = value
        _json['responses'] = responses

        # demographics
        _demographics = _json['demographics'].split('___')
        demographics = {}
        for elem in _demographics:
            key, value = elem.split('=')
            demographics[key] = value
        _json['demographics'] = demographics

        result.append(_json)

    json.dump(result, open('face-rating-data.json', 'w'), indent=4, separators=(',', ': '))

if __name__ == '__main__':
    guess_who_parser()
    face_rating()
