import re
import requests
import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


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
    

    def tokenize(self, text):
        return list(text)


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
    

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    

    def to_tokens(self, indices):
        if hasattr(indices, "__len__") and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]


    @property
    def unk(self):
        return self.token_to_idx["<unk>"]


def calculate_word_freq_indices(vocab_object):
    word_frequency = []
    token_position = []
    tokens = []
    token_position_count = 1
    for token, freq in vocab_object.token_freqs:
        word_frequency.append(freq)
        tokens.append(token)
        token_position.append(token_position_count)
        token_position_count += 1

    word_frequency_log = [np.log(wf) for wf in word_frequency]
    token_position_log = [np.log(tp) for tp in token_position]

    return word_frequency_log, token_position_log


def ziphs_law_plot(vocab_object, label, plot_name):
    if not isinstance(vocab_object, list):
        word_frequency_log, token_position_log = calculate_word_freq_indices(
                                                vocab_object)        

        plt.plot(token_position_log, word_frequency_log, label=label)
        plt.ylabel('frequency: n(x)')
        plt.xlabel('token: x')
        if label is not None:
            plt.legend()
        plt.savefig(plot_name)
    else:
        for v, l, p in zip(vocab_object, label, plot_name):
            ziphs_law_plot(v, l, p)

    
def calculate_params_for_zipf(vocab_objects):
    for vocab_label, vocab_object in vocab_objects.items():
        word_frequency_log, token_position_log = calculate_word_freq_indices(
                                                vocab_object)
        X = np.reshape(np.array(word_frequency_log), (-1,1))
        y = np.array(token_position_log)
        lr = LinearRegression()
        lr.fit(X=X, y=y)
        print(f"Coefficient for {vocab_label} is : {lr.coef_}")



if __name__ == "__main__":

    dpp = DataPreprocessingPipeline()

    data_url = "http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"
    data_dump_path = "data.txt"
    dpp.download_data(data_url, data_dump_path)

    text = dpp.read_file_contents(data_dump_path)
    print(f"\n==> Raw text contents without any preprocessing:\n{text[:100]}")
    
    text = dpp.preprocess(text)
    print(f"\nText after removal of non alphabets:\n{text[:100]}")

    tokens = dpp.tokenize(text)
    print(f"\ntext after tokenization:\n{','.join(tokens[:100])}")

    vocab = Vocabulary(tokens)
    indices = vocab[tokens[:10]]
    print(f"\nindices:\n{indices}")
    print(f"\nwords:\n{vocab.to_tokens(indices)}")

    # Language statistics word level for unigrams
    words = text.split()
    unigram_vocab = Vocabulary(words)
    print(f"\nTop 10 token frequencies:\n{unigram_vocab.token_freqs[:10]}")
    ziphs_law_plot(vocab_object=unigram_vocab, label=None,
                   plot_name="zipfs_law_word_freq_log_idx_log_unigram.jpg")
    
    # Language statistics word level for bigrams
    bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
    bigram_vocab = Vocabulary(bigram_tokens)
    print(f"\nBigram words token frequency:\n{bigram_vocab.token_freqs[:10]}")

    # Language statistics word level for trigrams
    trigram_tokens = ['--'.join(pair) for pair in zip(words[:-2], words[1:-1], 
                                                      words[2:])]
    trigram_vocab = Vocabulary(trigram_tokens)
    print(f"\nTrigram words token frequency:\n{trigram_vocab.token_freqs[:10]}")
    
    ziphs_law_plot(vocab_object=[unigram_vocab, bigram_vocab, trigram_vocab], 
                   label=["unigram", "bigram", "trigram"], 
                   plot_name=["zipfs_law_word_freq_log_idx_log.jpg"]*3)
    

    ### Excercise ###
    # 1. In the experiment of this section, tokenize text into words and vary 
    # the min_freq argument value of the Vocab instance. Qualitatively 
    # characterize how changes in min_freq impact the size of the resulting 
    # vocabulary.
    min_freqs = [1,2,3,4,5]
    for min_freq in min_freqs:
        words = text.split()
        vocab = Vocabulary(words, min_freq=min_freq)
        print(f"Vocab size for min frequency: {min_freq} is {len(vocab)}")

    # 2. Estimate the exponent of Zipfian distribution for unigrams, bigrams, 
    # and trigrams in this corpus.
    params = {
        "unigram": unigram_vocab,
        "bigram": bigram_vocab,
        "triigram": trigram_vocab,
    }
    calculate_params_for_zipf(vocab_objects=params)

    # 3. Find some other sources of data (download a standard machine learning 
    # dataset, pick another public domain book, scrape a website, etc). For each, 
    # tokenize the data at both the word and character levels. How do the 
    # vocabulary sizes compare with The Time Machine corpus at equivalent values 
    # of min_freq. Estimate the exponent of the Zipfian distribution 
    # corresponding to the unigram and bigram distributions for these corpora. 
    # How do they compare with the values that you observed for 
    # The Time Machine corpus?
    
    # The haunter of the dark by H. P. Lovecraft from 
    # https://www.gutenberg.org/ebooks/73233
    link = "https://www.gutenberg.org/cache/epub/73233/pg73233.txt"
    data_dump_path = "data_ex3.txt"
    dpp.download_data(link, data_dump_path)
    text = dpp.read_file_contents(data_dump_path)
    text = dpp.preprocess(text)

    words = text.split()
    unigram_vocab = Vocabulary(words)
    bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
    bigram_vocab = Vocabulary(bigram_tokens)

    min_freqs = [1,2,3,4,5]
    for min_freq in min_freqs:
        words = text.split()
        vocab = Vocabulary(words, min_freq=min_freq)
        print(f"Vocab size for min frequency: {min_freq} is {len(vocab)}")

    params = {
        "unigram": unigram_vocab,
        "bigram": bigram_vocab,
    }
    calculate_params_for_zipf(vocab_objects=params)