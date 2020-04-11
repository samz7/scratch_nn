from layer import NNlayer

"""
TODO:

    Figure out why the fuck the loss isn't going down
    probably the cost derivative is wrong
    maybe dw calculation is wrong
    
"""


"""
activations = ['r', 'r', 'r', 'r']
layer dims = (4, 5, 7, 4, 3)
            w[4, 5], w[5, 7], w[7, 4], w[4 , 3]
"""


class NNModel():
    def __init__(self, layer_dims, activations):
        self.nlayers = len(layer_dims);
        self.activations = activations; # len(activations) will be nlayers-1 cuz the input layer.
        
        self.params = {"l"+str(i+1):NNlayer(layer_dims[i], layer_dims[i+1], activations[i])
                            for i in range(len(layer_dims)-1)}
        
    
    def forward_propogate(self, x):
        for i in self.params:
            x = self.params[i].forward(x)
        return x;
    
    def backward_propogate(self, loss_derivative): # loss_derivative -> DJ/DAL(Activation of the last layer)
        loss_derivative = loss_derivative
        for i in reversed(range(1, self.nlayers)):
            loss_derivative = self.params['l'+str(i)].backward(loss_derivative)
    
    def update_model(self, learning_rate):
        for i in range(1,self.nlayers):
            self.params['l'+str(i)].update(learning_rate);
    
    def __repr__(self):
        ret = ''
        for i in range(1,self.nlayers):
            ret += self.params['l'+str(i)].__repr__() + '\n';
        return ret
            
