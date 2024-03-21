import jax
import math
import numpy as np
from random import shuffle
from jax import numpy as jnp
import matplotlib.pyplot as plt


# Implementing a simple DataLoader
class DataLoader(object):
    """
    Dataloader implementation from scratch
    """
    def __init__(self, X, y, tau, batch_size, train_test_split=0.2, 
                shuffle=True, *args, **kwargs):
        self.X = X
        self.y = y
        self.tau = tau
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.shuffle = shuffle
        self.data_size = self.y.shape[0]
        self.indices = list(np.arange(0, self.data_size, 1))

    
    def prepare_dataset(self):
        if self.train_test_split:
            self.test_size = int(self.data_size * self.train_test_split)
            self.train_size = self.data_size - self.test_size
            self.test_indices = self.indices[-self.test_size:][:-self.tau]
            self.train_indices = self.indices[:self.train_size][:-self.tau]
            self.test_indices_queue = list(self.test_indices)
            self.train_indices_queue = list(self.train_indices)
            self.train_steps = math.ceil(len(self.train_indices_queue)/batch_size)
            self.test_steps = math.ceil(len(self.test_indices_queue)/batch_size)
            return (self.train_steps, self.test_steps)
    

    def compile_batch(self, indices):
        x_batch, y_batch = [], []
        for index in indices:
            ts = self.X[index:index+self.tau+1]
            x_batch.append(ts[:self.tau])
            y_batch.append(ts[-1])
        return (jnp.asarray(x_batch), jnp.asarray(y_batch), jnp.asarray(indices))


    def train_iter(self):
        if self.shuffle:
            shuffle(self.train_indices_queue)

        if len(self.train_indices_queue) == 0:
            self.train_indices_queue = self.train_indices.copy()

        next_train_indices = [self.train_indices_queue.pop() for _ in 
                    range(min(self.batch_size, len(self.train_indices_queue)))]
        batch = self.compile_batch(next_train_indices)
        return batch


    def test_iter(self):
        if self.shuffle:
            shuffle(self.test_indices_queue)

        if len(self.test_indices_queue) == 0:
            self.test_indices_queue = self.test_indices.copy()
        next_test_indices = [self.test_indices_queue.pop() for _ in 
                    range(min(self.batch_size, len(self.test_indices_queue)))]

        batch = self.compile_batch(next_test_indices)
        return batch
    

    def sample_test_data(self, hop):
        valid_idxs_list = []
        self.train_indices_queue = self.train_indices.copy()
        self.test_indices_queue = self.test_indices.copy()
        for hop_num in range(int(self.data_size/hop)):
            idx_value = hop_num*hop
            if idx_value in self.train_indices_queue or idx_value in self.test_indices_queue:
                valid_idxs_list.append(idx_value)
        batch = self.compile_batch(valid_idxs_list)
        return batch


class LinearRegression(object):
    """
    Linear Regression implementation from scratch.
    """
    def __init__(self, inp_size, learning_rate=0.01):
        self.inp_size = inp_size
        self.learning_rate = learning_rate
        # Initialize weights and biases
        self.weights = jax.random.truncated_normal(
            key=jax.random.PRNGKey(0),
            lower = 0,
            upper = 1,
            shape=[inp_size],
            dtype=jnp.float32,
        ) * 0.01

        self.bias = jax.random.truncated_normal(
            key=jax.random.PRNGKey(0),
            lower = 0,
            upper = 1,
            shape=[1],
            dtype=jnp.float32,
        ) * 0.01

    
    def forward_pass(self, x_batch):
        self.out = jnp.matmul(x_batch, self.weights) + self.bias

    
    def backpropagation(self, x_batch, y_true):
        batch_size = y_true.shape[0]
        self.optimize(x_batch, y_true, self.out, batch_size)
        self.weights = self.weights - self.learning_rate * self.grad_w
        self.bias = self.bias - self.learning_rate * self.grad_b
        return self.weights, self.bias, self.error
    

    def calculate_error(self, y_true, y_pred):
        self.error = y_true - y_pred


    def optimize(self, x_batch, y_true, y_pred, batch_size):
        self.calculate_error(y_true, y_pred)
        self.grad_w = - 2 * jnp.matmul(self.error, x_batch) / batch_size
        self.grad_b = - 2 * jnp.sum(self.error) / batch_size
    

    def train(self, x_inp, y_gt):
        self.forward_pass(x_inp)
        weights, biases, error = self.backpropagation(x_inp, y_gt)
        return weights, biases, error, self.out


    def test(self, x_inp, y_gt):
        self.forward_pass(x_inp)
        self.calculate_error(y_gt, self.out)
        return self.out, self.error


    def multi_step_prediction(self, batch, num_preds):
        x_inp, y_gt, idxs = batch
        indexes = list(np.array(idxs))
        predictions = []
        for num_pred in range(num_preds):
            self.test(x_inp, y_gt)
            if  num_pred > 0:
                idxs += 1
                indexes += list(np.array(idxs))
            x_inp = jnp.concatenate((x_inp[:,1:], self.out.reshape(self.out.shape[0], -1)), axis=1)
            predictions += list(np.array(self.out))
        return predictions, indexes



if __name__ == "__main__":

    ### Creating synthetic Dataset
    batch_size = 16
    T = 1000
    num_train = 600
    tau = 4
    # Dataset is sin + noise for each time step
    time = np.arange(1, T+1, dtype=np.float32)
    x = np.sin(0.01 * time) + np.random.normal(loc=0.0, scale=1.0, 
                                               size=[T]) * 0.2
    fig1 = plt.figure(1)
    plt.scatter(time, x, color="blue", label="data")
    


    ### Training
    epochs = 5
    lr = 0.005
    
    dl = DataLoader(X=x, y=time, tau=5, batch_size=16, train_test_split=0.5)
    train_steps, test_steps = dl.prepare_dataset()
    lr = LinearRegression(inp_size=5, learning_rate=lr)
    no_train_steps, no_test_steps = dl.prepare_dataset()
    best_epoch = 0
    min_error = 0

    for epoch in range(epochs):
        error_acc_train, error_acc_test, pred_train, time_index_train, pred_test, time_index_test = [], [], [], [], [], []
        print(f"==> Running epoch : {epoch+1} / {epochs}")
        for j in range(no_train_steps):
            x_inp, y_gt, idxs = dl.train_iter()
            weight, bias, error, y_out = lr.train(x_inp ,y_gt)
            error_acc_train.append(np.array(jnp.sum(error)))
            pred_train += list(np.array(y_out))
            time_index_train += list(np.array(idxs))
        train_error = np.abs(np.sum(np.array(error_acc_train))/len(error_acc_train))
        print(f"Train error: {train_error}")

        for k in range(no_test_steps):
            x_inp, y_gt, idxs = dl.test_iter()
            y_out, error = lr.test(x_inp ,y_gt)
            error_acc_test.append(np.array(jnp.sum(error)))
            pred_test += list(np.array(y_out))
            time_index_test += list(np.array(idxs))
        test_error = np.abs(np.sum(np.array(error_acc_test))/len(error_acc_test))

        if epoch == 0:
            min_error = test_error
            best_epoch = epoch
        if min_error >= test_error:
            min_error = test_error
            best_epoch = epoch
        print(f"Test error: {test_error}")

    print(f"\nBest epoch is {best_epoch}/{epochs} with error: {min_error}")
    plt.scatter(time_index_train, pred_train, color="green", label="pred_train")
    plt.scatter(time_index_test, pred_test, color="red", label="pred_test")
    plt.legend() 
    plt.savefig("jax_data.jpg")
    plt.close()


    # Multistep prediction
    # Sample data for 1, 4, 16, 64 step predcitions so that we get a even plot
    step_preds = [1, 4, 16, 64]
    step_preds_color = {1:"yellow", 4:"red", 16:"green", 64:"cyan"}
    fig2 = plt.figure(2)
    plt.scatter(time, x, color="blue", label="data")
    for step_pred in step_preds:
        print(f"Running for step pred: {step_pred}")
        batch = dl.sample_test_data(hop=step_pred)
        predictions, indexes = lr.multi_step_prediction(batch, step_pred)
        plt.scatter(indexes, predictions, color=
            f"{step_preds_color[step_pred]}", label=f"{step_pred}-step preds")
    plt.legend()
    plt.savefig("jax_step_pred.jpg")
    plt.close()