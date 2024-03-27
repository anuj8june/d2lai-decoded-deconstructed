import re
import random
import requests
import numpy as np
import collections


class Vocabulary(object):
    """
    Vocabulary class for handling tokens, coversion to index and back.
    """
    def __init__(self, min_freq=0, reserved_tokens=[], *args, **kwargs):
        self.min_freq = min_freq
        self.reserved_tokens = reserved_tokens    
        
    def prepare_vocabulary(self, tokens):
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], 
                                  reverse=True)
        self.idx_to_token = list(sorted(set(["<unk>"] + self.reserved_tokens + [
            token for token, freq in self.token_freqs 
            if freq >= self.min_freq])))
        self.token_to_idx = {token: idx for idx, token in 
                             enumerate(self.idx_to_token)}


    def __len__(self):
        return len(self.idx_to_token)
    

    def token_to_indices(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.token_to_indices(token) for token in tokens]
    

    def indices_to_tokens(self, indices):
        if hasattr(indices, "__len__") and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]


    @property
    def unk(self):
        return self.token_to_idx["<unk>"]


class Dataloader(Vocabulary):
    """
    Dataloader for training, validation and testing.
    """
    def __init__(self, data_link: str, dump_path: str, batch_size: int, 
                 train_val_split: float) -> None:
        Vocabulary.__init__(self)
        self.data_link = data_link
        self.dump_path = dump_path
        self.batch_size = batch_size
        self.train_val_split = train_val_split


    def download_data(self):
        try:
            r = requests.get(self.data_link)
            with open(self.dump_path, "wb") as f:
                f.write(r.content)
                print(f"File downloaded from link: {self.data_link} and saved to path: {self.dump_path}")
        except Exception as e:
            print(e)


    def read_file_contents(self):
        try:
            with open(self.dump_path, "r") as f:
                self.text = f.read()
        except Exception as e:
            print(e)


    def preprocess(self):
        self.text = re.sub("[^A-Za-z]+", " ", self.text).lower()
    

    def char_tokenize(self):
        self.text = list(self.text)
    

    def prepare_data(self):
        self.download_data()
        self.read_file_contents()
        self.preprocess()
        self.char_tokenize()
        Vocabulary.prepare_vocabulary(self, self.text)

        len_text = len(self.text)
        total_samples = len_text - self.batch_size - 1
        self.num_test_samples = int(total_samples * self.train_val_split)
        self.num_train_samples = total_samples - self.num_test_samples
        total_index = list(np.arange(total_samples))
        random.shuffle(total_index)
        self.train_indices = total_index[:self.num_train_samples]
        self.test_indices = total_index[-self.num_test_samples:]
        self.train_indices_queue = self.train_indices.copy()
        self.test_indices_queue = self.test_indices.copy()
        return self.text, self.num_train_samples, self.num_test_samples


    def compile_batch(self, selected_indices: list):
        batch_X, batch_y, token_X, token_y = [], [], [], []
        print(selected_indices, type(selected_indices))
        for index in selected_indices:
            tokens = self.text[index:index+self.batch_size+1]
            one_hot_indices = Vocabulary.token_to_indices(self, tokens)
            batch_X.append(one_hot_indices[:-1])
            batch_y.append(one_hot_indices[-1])
            token_X.append(''.join(tokens[:-1]))
            token_y.append(tokens[-1])
        return np.array(batch_X), np.array(batch_y), token_X, token_y, selected_indices


    def next_train(self):
        if len(self.train_indices_queue) == 0:
            self.train_indices_queue = self.train_indices.copy()

        selected_indices = []
        for _ in range(min(self.batch_size, len(self.train_indices_queue))):
            selected_indices.append(self.train_indices_queue.pop())
        data_batch = self.compile_batch(selected_indices)
        return data_batch


    def next_test(self):
        if len(self.test_indices_queue) == 0:
            self.test_indices_queue = self.test_indices.copy()

        selected_indices = []
        for _ in range(min(self.batch_size, len(self.test_indices_queue))):
            selected_indices.append(self.test_indices_queue.pop())
        data_batch = self.compile_batch(selected_indices=selected_indices)
        return data_batch


class RNNCell(object):
    """
    RNN(character encoded) model implemented from scratch.
    """
    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Hidden layers
        self.w_xh = np.random.randn(vocab_size, hidden_dim)
        self.w_hh = np.random.randn(hidden_dim, hidden_dim)
        self.b_h = np.zeros(hidden_dim)
        
        # Output layers
        self.w_oh = np.random.randn(hidden_dim, vocab_size)
        self.b_o = np.random.randn(vocab_size)

    
    def __call__(self, inputs: np.array, prev_state : np.array=None
                 ) -> np.array:
        current_state = np.tanh(np.matmul(inputs, self.w_xh) + 
                        np.matmul(prev_state, self.w_hh) + self.b_h)
        current_output = np.matmul(current_state, self.w_oh) + self.b_o

        return current_output, current_state


class Trainer(Dataloader, RNNCell):
    """
    Trainer module for RNN implemented from scratch.
    """
    def __init__(self, data_link: str, dump_path: str, batch_size: int, 
                 train_val_split: float, vocab_size: int, hidden_dim: int, 
                 epochs: int, seq_len: int, learning_rate:float = 0.01
                 ) -> None:
        Dataloader.__init__(self, data_link=data_link, dump_path=dump_path, batch_size=batch_size, 
                            train_val_split=train_val_split)
        tokens, train_steps, test_steps = Dataloader.prepare_data(self)
        RNNCell.__init__(self, vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.epochs = epochs
        self.seq_len = seq_len
        self.learning_rate = learning_rate


    def forward_pass(self):
        self.out = self.model(self.train_X)


    def backpropagation(self):
        self.calculate_error_and_optimize()


    def calculate_error(self):
        self.error = self.out - self.train_y


    def train_step(self, seq_no):
        self.train_X = self.train_X()[:,seq_no,:].reshape[:,:]
        self.forward_pass()
        self.backpropagation()


    def val_step(self, seq_no):
        self.train_X = self.train_X()[:,seq_no,:].reshape[:,:]
        self.forward_pass()
        self.calculate


    def train(self):
        for epoch in self.epochs:
            print(f"\n===> Currently training epoch {epoch+1} / {self.epochs}")
            for seq_no in range(self.seq_len):
                self.train_X, self.train_y = self.dataloader.next_train()
                self.val_X, self.val_y = self.dataloader.next_test()
                self.train_step(seq_no)
                self.val_step(seq_no)


    @staticmethod
    def softmax(x):
        """
        Compute softmax values for each sets of scores in x.
        """
        e_x = np.exp(x)
        return e_x / np.sum(e_x)


if __name__ == "__main__":

    # Testing Dataloader class
    data_link = "http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"
    dump_path = "data.txt"
    batch_size = 8
    train_val_split = 0.1
    dl = Dataloader(data_link=data_link, dump_path=dump_path, 
                    batch_size=batch_size, train_val_split=train_val_split)
    text, num_train_samples, num_test_samples = dl.prepare_data()
    print(f"\nNum train samples: {num_train_samples} | Num test samples: {num_test_samples}")
    for i in range(num_train_samples):
        print(i)
        X, y, lX, ly, si = dl.next_train()
        print(X.shape, y.shape, lX, ly, si)
        if i > 3:
            break

    for i in range(num_test_samples):
        print(i)
        X, y, lX, ly, si = dl.next_test()
        print(X.shape, y.shape, lX, ly, si)
        if i > 3:
            break

    # Testing RNNCell class
    vocab_size, hidden_dim = 26, 32
    inputs = np.random.randn(batch_size, vocab_size)
    prev_state = np.random.randn(batch_size, hidden_dim)
    print(f"\nInput shape: {inputs.shape} | Prev state: {prev_state.shape}")

    rnn = RNNCell(vocab_size=vocab_size, hidden_dim=32)
    outputs, state = rnn(inputs, prev_state)
    print(f"Outputs:{outputs.shape}, States:{state.shape}")

    # Training parameters
    epochs, seq_len, lr, train_val_split = 5, 16, 0.01, 0.1
    # trainer = Trainer(data_link=data_link, dump_path=dump_path, 
    #                   batch_size=batch_size, train_val_split=train_val_split, 
    #                   vocab_size=vocab_size, hidden_dim=32, epochs=epochs, 
    #                   seq_len=seq_len, learning_rate=lr)
    # # trainer.train()