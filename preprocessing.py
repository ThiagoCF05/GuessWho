__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 14/08/2018
Description:
    Preprocess GuessWho for the QA task, providing a train, dev and test split, as well as a vocabulary
"""

import json
import os
import numpy as np
import random

from scipy import misc

CORPUS_PATH = 'dataset/guess-who-data.json'
QA_TASK_PATH = 'dataset/qa_task'
IMAGE_PATH = 'dataset/images'
PROC_IMAGE_PATH = 'dataset/procimages'

def load_corpus():
    path = CORPUS_PATH
    corpus = json.load(open(path))
    return corpus

def prepare_dataset(corpus):
    dataset = []

    participants = list(set(map(lambda x: x['uniqueID'], corpus)))
    for participant in participants:
        participant_games = list(filter(lambda x: x['uniqueID'] == participant, corpus))
        boardNumbers = sorted(list(set(map(lambda x: x['boardNumber'], participant_games))))

        for boardNumber in boardNumbers:
            questions = sorted(filter(lambda x: x['boardNumber'] == boardNumber, participant_games), key=lambda x:x['questionNumber'])
            prev_noFaces = []
            for question in questions:
                print('Participant: ', participant, 'Game number: ', boardNumber, 'Question: ', question['questionNumber'])
                q = ' '.join(question['tokens'])
                noFaces = list(filter(lambda x: x not in prev_noFaces and x != '', question['eliminatedFaces']))
                yesFaces = list(filter(lambda x: x not in question['eliminatedFaces'] and x != '', question['allFaces']))
                prev_noFaces = question['eliminatedFaces']

                for noFace in noFaces:
                    dataset.append({'face_id':noFace, 'question':q, 'answer':'no'})
                for yesFace in yesFaces:
                    dataset.append({'face_id':yesFace, 'question':q, 'answer':'yes'})
    return dataset

def split(dataset):
    images = list(set(map(lambda x: x['face_id'], dataset)))
    sample_size = len(images) * 0.1
    random.shuffle(images)

    test_images = images[:int(sample_size)]

    testset = list(filter(lambda x: x['face_id'] in test_images, dataset))
    auxset = list(filter(lambda x: x['face_id'] not in test_images, dataset))

    sample_size = int(len(dataset) * 0.1)
    random.shuffle(auxset)
    devset = auxset[:sample_size]
    trainset = auxset[sample_size:]

    print('Dataset size: ', len(dataset))
    print('Trainset size: ', len(trainset))
    print('Devset size: ', len(devset))
    print('Testset size: ', len(testset))
    return trainset, devset, testset

def vocabulary(trainset):
    vocabulary = []
    for row in trainset:
        question = row['question'].strip().lower().split()
        vocabulary.extend(question)

    print('Frequency: ', len(vocabulary))
    vocabulary = list(set(vocabulary))
    vocabulary.append('unk')
    vocabulary.append('eos')
    print('Vocabulary: ', len(vocabulary))

    word2id = dict([(word, i) for i, word in enumerate(vocabulary)])
    id2word = dict(map(lambda x: (x[1], x[0]), word2id.items()))
    return word2id, id2word

def proc_image():
    # load data
    trainset, devset, testset, word2id, id2word = load_data()

    # computing mean
    images = []
    for row in trainset:
        img = os.path.join(IMAGE_PATH, row['face_id']+'.JPG')
        image = misc.imread(img)
        image = image[100:image.shape[0]-100,:] # crop
        images.append(image)
    mean = np.mean(np.array(images), axis=0)
    del images

    # create directory
    if not os.path.exists(PROC_IMAGE_PATH):
        os.mkdir(PROC_IMAGE_PATH)

    imgs = os.listdir(IMAGE_PATH)
    for img in imgs:
        path = os.path.join(IMAGE_PATH, img)
        image = misc.imread(path)
        image = image[100:image.shape[0]-100,:] # crop
        image = image - mean
        misc.imsave(os.path.join(PROC_IMAGE_PATH, img), image)

def generate():
    # load corpus
    corpus = load_corpus()
    # prepare dataset
    dataset = prepare_dataset(corpus)
    # split dataset in training, dev and test set making sure to have a unique set of images in testset
    trainset, devset, testset = split(dataset)
    # process images cropping and subtracting the train mean

    word2id, id2word = vocabulary(trainset)

    # SAVE
    # saving train, dev and test splits
    path = QA_TASK_PATH
    if not os.path.exists(path):
        os.mkdir(path)
    json.dump(trainset, open(os.path.join(path, 'train.json'), 'w'), separators=(',', ':'), indent=4)
    json.dump(devset, open(os.path.join(path, 'dev.json'), 'w'), separators=(',', ':'), indent=4)
    json.dump(testset, open(os.path.join(path, 'test.json'), 'w'), separators=(',', ':'), indent=4)

    json.dump(word2id, open(os.path.join(path, 'word2id.json'), 'w'), separators=(',', ':'), indent=4)
    json.dump(id2word, open(os.path.join(path, 'id2word.json'), 'w'), separators=(',', ':'), indent=4)

def load_data():
    path = QA_TASK_PATH
    trainset = json.load(open(os.path.join(path, 'train.json')))
    devset = json.load(open(os.path.join(path, 'dev.json')))
    testset = json.load(open(os.path.join(path, 'test.json')))

    word2id = json.load(open(os.path.join(path, 'word2id.json')))
    id2word = json.load(open(os.path.join(path, 'id2word.json')))
    return trainset, devset, testset, word2id, id2word

if __name__ == '__main__':
    if not os.path.exists(QA_TASK_PATH):
        generate()