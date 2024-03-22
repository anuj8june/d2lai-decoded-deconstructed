import re
import requests
import collections


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

    