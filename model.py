__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 14/08/2018
Description:
    Model for GuessWho question answering
"""

import dynet as dy
import preprocessing
import os
import numpy as np
import time

from scipy import misc

class QA():
    def __init__(self):
        print('READING CORPUS...')
        self.trainset, self.devset, self.testset, self.word2id, self.id2word = preprocessing.load_data()
        self.img_path = 'dataset/images'
        print('PREPARING CORPUS...')
        self.id2img = {}
        self.trainset = self.prepare_corpus(self.trainset)
        self.devset = self.prepare_corpus(self.devset)
        print('SUBTRACTING THE MEAN...')
        self.subtract_mean()

        self.UNK = 'unk'
        self.EOS = 'eos'

        print('INITIALIZING MODEL...')
        self.EPOCH = 300
        self.BATCH = 32

        dy.renew_cg()
        self.model = dy.Model()
        self.init_convnet()
        self.init_lstm()

    def prepare_corpus(self, dataset, testset=False):
        _dataset = []
        for row in dataset:
            question = row['question'].lower().split()
            if len(question) != 0:
                answer = 1 if row['answer'] == 'yes' else 0
                _dataset.append({'question':question, 'answer':answer, 'face_id':row['face_id']})

            if row['face_id'] not in self.id2img and not testset:
                img = os.path.join(self.img_path, row['face_id']+'.JPG')
                image = misc.imread(img)
                image = image[100:image.shape[0]-100,:]
                self.id2img[row['face_id']] = image
        return _dataset

    def subtract_mean(self):
        values = np.array(list(self.id2img.values()))
        images = np.concatenate(values)
        mean = np.mean(images, axis=0)

        for img in self.id2img:
            self.id2img[img] = self.id2img[img] - mean

    def init_convnet(self):
        self.IMAGE_DIM = 512

        # CONVOLUTION
        self.RESHAPING = (13 * 13 * 512)

        self.F1 = self.model.add_parameters((5, 5, 3, 64))
        self.b1 = self.model.add_parameters((64, ))

        self.F2 = self.model.add_parameters((5, 5, 64, 128))
        self.b2 = self.model.add_parameters((128, ))

        self.F31 = self.model.add_parameters((5, 5, 128, 256))
        self.b31 = self.model.add_parameters((256, ))
        # self.F32 = self.model.add_parameters((3, 3, 128, 256))
        # self.b32 = self.model.add_parameters((256, ))

        self.F41 = self.model.add_parameters((5, 5, 256, 512))
        self.b41 = self.model.add_parameters((512, ))
        # self.F42 = self.model.add_parameters((3, 3, 256, 512))
        # self.b42 = self.model.add_parameters((512, ))

        self.F51 = self.model.add_parameters((5, 5, 512, 512))
        self.b51 = self.model.add_parameters((512, ))
        # self.F52 = self.model.add_parameters((3, 3, 512, 512))
        # self.b52 = self.model.add_parameters((512, ))

        input_size = self.RESHAPING
        self.W1 = self.model.add_parameters((self.IMAGE_DIM, input_size))
        self.bW1 = self.model.add_parameters((self.IMAGE_DIM))

        # self.W2 = self.model.add_parameters((self.IMAGE_DIM, self.IMAGE_DIM))
        # self.bW2 = self.model.add_parameters((self.IMAGE_DIM))

    def convnet(self, image):
        x = dy.inputTensor(image)

        x = dy.conv2d_bias(x, self.F1, self.b1, [1, 1], is_valid=False)
        x = dy.maxpooling2d(x, [2, 2], [2, 2], is_valid=False)
        x = dy.rectify(x)

        x = dy.conv2d_bias(x, self.F2, self.b2, [1, 1], is_valid=False)
        x = dy.maxpooling2d(x, [2, 2], [2, 2], is_valid=False)
        x = dy.rectify(x)

        x1 = dy.conv2d_bias(x, self.F31, self.b31, [1, 1], is_valid=False)
        x1 = dy.maxpooling2d(x1, [2, 2], [2, 2], is_valid=False)
        x1 = dy.rectify(x1)

        # x2 = dy.conv2d_bias(x, self.F32, self.b32, [1, 1], is_valid=False)
        # x2 = dy.maxpooling2d(x2, [2, 2], [2, 2], is_valid=False)

        x1 = dy.conv2d_bias(x1, self.F41, self.b41, [1, 1], is_valid=False)
        x1 = dy.maxpooling2d(x1, [2, 2], [2, 2], is_valid=False)
        x1 = dy.rectify(x1)

        # x2 = dy.conv2d_bias(x2, self.F42, self.b42, [1, 1], is_valid=False)
        # x2 = dy.maxpooling2d(x2, [2, 2], [2, 2], is_valid=False)

        x1 = dy.conv2d_bias(x1, self.F51, self.b51, [1, 1], is_valid=False)
        x1 = dy.maxpooling2d(x1, [2, 2], [2, 2], is_valid=False)
        x1 = dy.rectify(x1)
        #
        # x2 = dy.conv2d_bias(x2, self.F52, self.b52, [1, 1], is_valid=False)
        # x2 = dy.maxpooling2d(x2, [2, 2], [2, 2], is_valid=False)

        x = dy.reshape(x1, (self.RESHAPING,))
        # x2 = dy.reshape(x2, (self.RESHAPING,))
        # x = dy.concatenate([x1, x2])

        vector = self.W1 * x + self.bW1
        # vector = self.W2 * vector + self.bW2
        return vector

    def init_lstm(self):
        LSTM_NUM_OF_LAYERS = 1
        EMBEDDING_SIZE = 512
        INPUT_DIM = self.IMAGE_DIM + EMBEDDING_SIZE
        HIDDEN_DIM = 1024
        INPUT_SIZE = len(self.word2id)
        OUTPUT_SIZE = 2

        self.lookup = self.model.add_lookup_parameters((INPUT_SIZE, EMBEDDING_SIZE))

        self.enc_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, INPUT_DIM, HIDDEN_DIM, self.model)
        self.enc_fwd_lstm.set_dropout(0.3)
        self.enc_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, INPUT_DIM, HIDDEN_DIM, self.model)
        self.enc_bwd_lstm.set_dropout(0.3)

        self.W = self.model.add_parameters((OUTPUT_SIZE, 2*HIDDEN_DIM))
        self.b = self.model.add_parameters((OUTPUT_SIZE,))

    def embed_question(self, question, image_conv):
        wordidx = []
        for w in question:
            try:
                wordidx.append(self.word2id[w])
            except:
                wordidx.append(self.word2id[self.UNK])

        return [dy.concatenate([self.lookup[word], image_conv]) for word in wordidx]

    def run_lstm(self, init_state, input_vecs):
        s = init_state

        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors

    def run(self, question, image):
        image_conv = self.convnet(image)

        embeddings = self.embed_question(question, image_conv)
        h0 = dy.concatenate([self.lookup[self.word2id[self.EOS]], image_conv])

        init_state = self.enc_fwd_lstm.initial_state().add_input(h0)
        fwd_vectors = self.run_lstm(init_state, embeddings)

        embeddings_rev = list(reversed(embeddings))
        init_state = self.enc_bwd_lstm.initial_state().add_input(h0)
        bwd_vectors = self.run_lstm(init_state, embeddings_rev)
        bwd_vectors = list(reversed(bwd_vectors))

        vector = dy.average([dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)])

        return dy.softmax(self.W * vector + self.b)

    def get_loss(self, image, question, answer):
        probs = self.run(question, image)

        return -dy.log(dy.pick(probs, answer))

    def validate(self):
        acc = 0
        for i, devrow in enumerate(self.devset):
            question = devrow['question']
            image = self.id2img[devrow['face_id']]
            answer = devrow['answer']

            probs = self.run(question, image).vec_value()
            pred = probs.index(max(probs))

            if pred == answer:
                acc += 1

            if i % self.BATCH == 0:
                dy.renew_cg()
        return float(acc) / len(self.devset)


    def train(self):
        trainer = dy.AdadeltaTrainer(self.model)

        epoch_timing = []
        early = 0.0
        best_acc = 0.0
        f = open('logging.txt', 'w')
        for epoch in range(self.EPOCH):
            print('\n')
            dy.renew_cg()
            losses = []
            closs = 0
            batch_timing = []
            for i, trainrow in enumerate(self.trainset):
                start = time.time()
                question = trainrow['question']
                answer = trainrow['answer']
                image = self.id2img[trainrow['face_id']]

                loss = self.get_loss(image, question, answer)
                losses.append(loss)
                end = time.time()
                t = (end-start)
                batch_timing.append(t)
                epoch_timing.append(t)

                if len(losses) == self.BATCH:
                    loss = dy.esum(losses)
                    _loss = loss.value()
                    closs += _loss
                    loss.backward()
                    trainer.update()
                    dy.renew_cg()

                    # percentage of trainset processed
                    percentage = str(round((float(i+1) / len(self.trainset)) * 100,2)) + '%'
                    # time of epoch processing
                    time_epoch = sum(epoch_timing)
                    if time_epoch > 3600:
                        time_epoch = str(round(time_epoch / 3600, 2)) + ' h'
                    elif time_epoch > 60:
                        time_epoch = str(round(time_epoch / 60, 2)) + ' min'
                    else:
                        time_epoch = str(round(time_epoch, 2)) + ' sec'

                    print("Epoch: {0} \t\t Loss: {1} \t\t Epoch time: {2} \t\t Trainset: {3}".format(epoch+1, round(_loss, 2), time_epoch, percentage), end='       \r')
                    losses = []
                    batch_timing = []

            print("\nEpoch: {0} \t\t Total Loss / Batch: {1}".format(epoch+1, round(closs / self.BATCH, 2)))
            acc = self.validate()
            print("\nEpoch: {0} \t\t Dev acc: {1} \t\t Best acc: {2}".format(epoch+1, round(acc,2), round(best_acc,2)))
            f.write("Epoch: {0} \t\t Dev acc: {1} \t\t Best acc: {2}\n".format(epoch+1, round(acc,2), round(best_acc,2)))
            if acc > best_acc:
                best_acc = acc
                early = 0
            else:
                early += 1

            if early == 50:
                break
            epoch_timing = []
        f.close()


if __name__ == '__main__':
    model = QA()
    print('TRAINING...')
    model.train()