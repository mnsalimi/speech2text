from os import path
from nltk import ngrams


class FiveGram:

    def __init__(self, data_address):
        self.train = open(path.join(data_address, "train.txt"), mode="r", encoding="utf-8").readlines()
        self.train = [line.strip().lstrip().rstrip() for line in self.train]
        self.test = open(path.join(data_address, "test.txt"))
        self.freqs = {}
        self.paddings_start = ["s1", "s2", "s3", "s4"]
        self.paddings_end = ["e1", "e2", "e3", "e4"]
        self.vocabs = []
        for line in self.train:
            line = line.split(" ")
            for tkn in line:
                if tkn not in self.vocabs:
                    self.vocabs.append(tkn)
        self.vocabs += self.paddings_start + self.paddings_end
        
    def create_ngrams_tokens(self):
        # self.fivegrams = []
        for line in self.train:
            line = line.split(" ")
            line[0:0] = self.paddings_start
            line[len(line):len(line)] = self.paddings_end
            fourgrams = ngrams(line, 4)
            fivegrams = ngrams(line, 5)
            for fivegram in fivegrams:
                self.freqs[' '.join(fivegram)] = self.freqs.get(fivegram, 0) + 1
            for fourgram in fourgrams:
                self.freqs[' '.join(fourgram)] = self.freqs.get(fourgram, 0) + 1

    def predict(self, text):
        text = text.split(" ")
        text[0:0] = self.paddings_start
        probs = {}
        for token in self.vocabs:
            probs[' '.join(text+[token])] = float(
                (1+self.freqs.get(' '.join(text[-4:])+" "+token, 0)) / (len(self.vocabs)+self.freqs[' '.join(text[-4:])])
            )
        return max(probs, key=probs.get).split(" ")[-1]


text = "i am moein salimi"
fivegram = FiveGram("data")
fivegram.create_ngrams_tokens()
x = fivegram.predict(text)
print(x)