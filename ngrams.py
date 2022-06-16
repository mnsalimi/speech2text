from os import path
from nltk import ngrams
from time import time
import pickle

class FiveGram:

    def __init__(self, data_address=None):
        # self.train = open(path.join(data_address, "train.txt"), mode="r", encoding="utf-8").readlines()
        # self.train = [line.strip().lstrip().rstrip() for line in self.train]
        # self.test = open(path.join(data_address, "test.txt"))
        # self.freqs = {}
        self.paddings_start = ["s1", "s2", "s3", "s4"]
        # self.paddings_end = ["e1", "e2", "e3", "e4"]
        # self.vocabs = []
        # for line in self.train:
        #     line = line.split(" ")
        #     for tkn in line:
        #         if tkn not in self.vocabs:
        #             self.vocabs.append(tkn)
        # self.vocabs += self.paddings_start + self.paddings_end
        # self.freqs += self.paddings_start + self.paddings_end
        with open("ngram_models/vocabs.pickle",'rb') as f:
            self.vocabs = pickle.load(f)
        with open("ngram_models/frequencies.pickle",'rb') as f:
            self.freqs = pickle.load(f)
        
    def create_ngrams_tokens(self):
        # self.fivegrams = []
        for line in self.train:
            line = line.split()
            line[0:0] = self.paddings_start
            line[len(line):len(line)] = self.paddings_end
            fourgrams = ngrams(line, 4)
            fivegrams = ngrams(line, 5)
            for fivegram in fivegrams:
                self.freqs[' '.join(fivegram)] = self.freqs.get(fivegram, 0) + 1
            for fourgram in fourgrams:
                self.freqs[' '.join(fourgram)] = self.freqs.get(fourgram, 0) + 1
        with open("ngram_models/frequencies.pickle", "wb") as f:
            pickle.dump(self.freqs, f)
        with open("ngram_models/vocabs.pickle", "wb") as f:
            pickle.dump(self.vocabs, f)

    def predict(self, text):
        text = text.split()
        pad_size = 0 if len(text)>4 else 4 - len(text)
        # print(pad_size)
        text[0:0] = self.paddings_start[:pad_size]
        # print(text)
        probs = {}
        for token in self.vocabs:
            probs[' '.join(text+[token])] = float(
                (1+self.freqs.get(' '.join(text[-4:])+" "+token, 0)) / (len(self.vocabs)+self.freqs.get(' '.join(text[-4:]), 0))
            )
        return ' '.join(text[5:] + [max(probs, key=probs.get).split()[-1]])

# t1 = time()
# text = "با"
# fivegram = FiveGram()
# # t2 = time()
# # print(t2-t1)
# # fivegram.create_ngrams_tokens()
# t3 = time()
# # print(t3-t2)
# x = fivegram.predict(text)
# t4 = time()
# print(x)
# print(t4-t3)