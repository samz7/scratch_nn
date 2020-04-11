import numpy as np

class NNlayer():
    def __init__(self, i_dim, o_dim, activation='relu'):
        self.i_dim = i_dim;
        self.o_dim = o_dim;
        self.activation_str = activation;
        self.w = np.random.randn(i_dim, o_dim);
        self.b = np.zeros((1, o_dim))
        self.activation = eval("self."+activation);
        
        #Derivatives
        self.dzdw = None;
        self.dadz = None;
        
        self.dw = None;
        self.db = None;
    

    
    
    def sigmoid(self, x, deriv=False):
        s = 1/(1+np.exp(-x));
        if deriv:
            return s*(1-s);
        return s;
    
    def relu(self, x, deriv=False):
        if deriv:
            return 1. * (x > 0);
        return np.maximum(x, 0, x);
    
    
    def tanh(self, x, deriv=False):
        t = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x));
        if deriv:
            return (1 - np.square(t))
        return t
    
    def forward(self, x):
        z = np.dot(x, self.w) + self.b;
        
        #Calculate intermidiate gradients
        self.dzdw = x.T #dervative of the weights with respect to the linear unit
        self.dadz = self.activation(z, deriv=True) # derivative of activations with respect to the linear unit
        
        a = self.activation(z);
        return a;
    
    def backward(self, grad_A): #grad_A is DJ/DA(where A is the previous activation layer)
        
        m = grad_A.shape[0]
        
        dz = self.dadz*grad_A # derivative of z with respect to the loss
        
        self.dw = np.dot(self.dzdw, dz)
        self.db = np.mean(dz, axis=0, keepdims=True);
        
        return np.dot(dz,self.w.T) # derivative of the next activation layer
    
    def update(self, learning_rate):
        self.w += learning_rate * -self.dw;
        self.b += learning_rate * -self.db
    
    def __call__(self, x):
        return self.forward(x);
    
    def __repr__(self):
        return "NNLayer(input_dim={}, output_dim={}, activation={})".format(self.i_dim, 
                                                                            self.o_dim, 
                                                                            self.activation_str)
    
        
