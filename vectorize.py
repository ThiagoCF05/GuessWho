__author__='thiagocastroferreira'

import h5py
import json
import os

from allennlp.commands.elmo import ElmoEmbedder

GAME_PATH = 'dataset/guess-who-data.json'
FACE_PATH='dataset/face-rating-data.json'

class Vectorize:
    def __init__(self):
        games = json.load(open(GAME_PATH))
        descriptions = json.load(open(FACE_PATH))

        path_ = '/roaming/tcastrof/drew'
        with open(os.path.join(path_, 'game.txt'), 'w') as f:
            f.write('\n'.join([' '.join(snt['tokens']) for snt in games]))

        with open(os.path.join(path_, 'face.txt'), 'w') as f:
            f.write('\n'.join([' '.join(snt['tokens']) for snt in descriptions]))

        self.elmo = ElmoEmbedder(cuda_device=1)

        vectors = [self.elmo.embed_sentence(snt['tokens']) for snt in games]
        path = os.path.join(path_, 'game_elmo.hdf5')
        with h5py.File(path, 'w') as hf:
            for i, vector in enumerate(vectors):
                hf.create_dataset(str(i),  data=vectors[i])

        vectors = [self.elmo.embed_sentence(snt['tokens']) for snt in descriptions]
        path = os.path.join(path_, 'face_elmo.hdf5')
        with h5py.File(path, 'w') as hf:
            for i, vector in enumerate(vectors):
                hf.create_dataset(str(i),  data=vectors[i])

if __name__ == '__main__':
    Vectorize()