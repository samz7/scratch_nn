import numpy as np

def softmax(x):
    exps = np.exp(x - np.max(x))
    
    return exps / exps.sum()

def cross_entropy_loss(y, y_hat):
    return -np.mean(y*np.log(y_hat + 1e-9))


def one_hotify(vector):
    """
    args:
    vector-> (m,1) dimensional categorical vector

    output:
    (m,C) dimensional matrix of one hot vectors
          where C is the number of unique values in vector

    """
    rows = np.arange(vector.shape[0])
    one_hot_v = np.zeros((vector.shape[0], len(np.unique(vector))))
    one_hot_v[rows, vector] = 1

    one_hot_v = np.flip(one_hot_v, axis=1);

    return one_hot_v;

def normalize(x):
    means = np.mean(x, axis=0, keepdims=True);
    deviations = x - means;
    return deviations / deviations ** 2

def train_model(model, loss_function, learning_rate, x_data, y_data, epochs=200, apply_softmax=True):
    for epoch in range(epochs):
        perm = np.random.permutation(x_data.shape[0])
        y_data = y_data[perm, :]
        x_data = x_data[perm, :]
        
        y_hat = model.forward_propogate(x_data);
        
        if apply_softmax:
            y_hat = softmax(y_hat);
        
        cost = loss_function(y_data, y_hat);
        print(cost);
        cost_derivative = y_hat - y_data;
        
        model.backward_propogate(cost_derivative);
        model.update_model(learning_rate);
    return model;
