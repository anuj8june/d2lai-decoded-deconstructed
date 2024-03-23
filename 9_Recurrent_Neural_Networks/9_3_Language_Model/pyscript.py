import re
import requests
import collections
import numpy as np


class DataPreprocessingPipeline(object):

    def __init__(self, *args, **kwargs):
        pass

    def download_data(self, link, dump_path):
        try:
            r = requests.get(link)
            with open(dump_path, "wb") as f:
                f.write(r.content)
                print(f"File downloaded from link: {link} and saved to path: {dump_path}")
        except Exception as e:
            print(e)


    def read_file_contents(self, fpath):
        try:
            with open(fpath, "r") as f:
                return f.read()
        except Exception as e:
            print(e)


    def preprocess(self, text):
        return re.sub("[^A-Za-z]+", " ", text).lower()
    

    def char_tokenize(self, text):
        return list(text)
    

    def word_tokenize(self, text):
        return text.split()


class Vocabulary(object):

    def __init__(self, tokens, min_freq=0, reserved_tokens=[], *args, **kwargs):
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], 
                                  reverse=True)
        self.idx_to_token = list(sorted(set(["<unk>"] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx for idx, token in 
                             enumerate(self.idx_to_token)}
        

    def __len__(self):
        return len(self.idx_to_token)
    

    def token_to_indexes(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.token_to_indexes(token) for token in tokens]
    

    def to_tokens(self, indices):
        if hasattr(indices, "__len__") and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]


    @property
    def unk(self):
        return self.token_to_idx["<unk>"]


class Dataloader(Vocabulary):
    def __init__(self, tokens, sample_len=20, min_freq=0, 
                 reserved_tokens=[], batch_size=2, num_steps=10, 
                 num_train=10000, num_val=5000, *args, **kwargs):
        super().__init__(tokens, min_freq=min_freq, 
                         reserved_tokens=reserved_tokens, *args, **kwargs)
        self.tokens = tokens
        self.batch_size = batch_size
        self.num_steps = min(num_steps, len(tokens)-2)
        self.num_train = num_train
        self.num_val = num_val
        self.current_position = 0
        self.sample_len = sample_len


    def next_data_batch(self):
        batch_X, batch_y, text_X, text_y = [], [], [], []
        for _ in range(self.batch_size):
            X = self.tokens[
                self.current_position:self.current_position+self.sample_len]
            y = self.tokens[
                self.current_position+1:self.current_position+self.sample_len+1]
            print(X,y)
            self.current_position += 1
            batch_X.append(Vocabulary.token_to_indexes(self, tokens=list(X)))
            batch_y.append(Vocabulary.token_to_indexes(self, tokens=list(y)))
            text_X.append(X)
            text_y.append(y)
        return np.array(batch_X), np.array(batch_y), text_X, text_y


if __name__ == "__main__":

    dpp = DataPreprocessingPipeline()

    data_url = "http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"
    data_dump_path = "data.txt"
    dpp.download_data(data_url, data_dump_path)

    text = dpp.read_file_contents(data_dump_path)
    print(f"\n==> Raw text contents without any preprocessing:\n{text[:100]}")
    
    text = dpp.preprocess(text)
    print(f"\nText after removal of non alphabets:\n{text[:100]}")

    tokens = dpp.char_tokenize(text)
    print(f"\ntext after tokenization:\n{','.join(tokens[:100])}")

    vocab = Vocabulary(tokens)
    indices = vocab.token_to_indexes(tokens[:10])
    print(f"\nindices:\n{indices}")
    print(f"\nwords:\n{vocab.to_tokens(indices)}")

    dl = Dataloader(text, sample_len=20, min_freq=0, reserved_tokens=[], batch_size=2, 
                 num_steps=10, num_train=10000, num_val=5000)
    
    for _ in range(5):
        x, y, xtxt, ytxt = dl.next_data_batch()
        print(f"\nX: {x.shape} | y: {y.shape} | Xtext: {xtxt} | ytxt:{ytxt}")


    ### Exercise ###
    # 1. Suppose there are 100,000 words in the training dataset. How much word 
    # frequency and multi-word adjacent frequency does a four-gram need to store?
    tokens = text.split()[:100000]
    tokens_quadgram = ['--'.join(pair) for pair in zip(tokens[:-3],tokens[1:-2],
                                           tokens[2:-1],tokens[3:])]
    quadgram_vocab = Vocabulary(tokens=tokens_quadgram)
    print(f"\nVocab size for quadgram: {len(tokens_quadgram)}")

    # 2. How would you model a dialogue?
    # To model a dialogues we also need to pass the information of each token(be
    # it string or word) along with the person who said it.
    # For example the following conversation between two person can be 
    # represented as:
    # Joe: Hi, how was your day?
    # Patrick: It was awesome.
    # [(Hi, Joe), (how,Joe), (was, Joe), (your,Joe), (day,Joe)]
    # [(It, Patrick), (was, Patrick), (awesome, Patrick)]
    # Both words and person saying them can be vetorized and concatenated and 
    # sent to model

    # 3. What other methods can you think of for reading long sequence data?
    # Reading data as subwords, Bag of words, count vectorizer

    # 4. Consider our method for discarding a uniformly random number of the 
    # first few tokens at the beginning of each epoch.
    # a. Does it really lead to a perfectly uniform distribution over the 
    #    sequences on the document?
        # On an average it will tend towards uniform distribution as words are 
        # uniformly removed at random.
    # b. What would you have to do to make things even more uniform?
        # Rather than doing it randomly we can do it more structurally and keep
        # track of what is removed and not removed.
    
    # 5. If we want a sequence example to be a complete sentence, what kind of 
    # problem does this introduce in minibatch sampling? How can we fix it?
    # The length of each sample will vary. We can fix it by introducing padding 
    # to some degree