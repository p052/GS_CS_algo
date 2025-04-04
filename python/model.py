import numpy as np
from keras.datasets import fashion_mnist # type: ignore
from keras.models import load_model # type: ignore
from keras.utils import to_categorical # type: ignore
from copy import deepcopy

class mnist_dataset:
    '''Loads Fashion MNIST dataset to evaluate an instance of model_pruning'''
    def __init__(self):
        ((X_train, y_train), (X_test, y_test)) = fashion_mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0],28,28,1)
        X_test = X_test.reshape(X_test.shape[0],28,28,1)

        self.X_train = X_train/256
        self.X_test = X_test/256

        self.y_train = to_categorical(y_train, 10)
        self.y_test = to_categorical(y_test, 10)

# Declares an instance that will be shared among model_pruning instances
mnist_instance = mnist_dataset()

class model_pruning:
    '''model_pruning represents a NN M whose original weights are saved in path.
       several functions are used to evaluate M on a mnist_instance, prune M at threshold, update its weights
    '''
    def __init__(self, path, threshold=0.01):
        self.threshold = threshold
        self.model = load_model(path)
        self.original_weights = deepcopy(self.model.get_weights())
        self.current_weights = deepcopy(self.original_weights)
        self.current_weights_pruned = None
        self.percentage_pruned = None
        self.prune()
    
    def evaluate(self):
        self.model.set_weights(self.current_weights)
        return(np.round(self.model.evaluate(mnist_instance.X_test, mnist_instance.y_test, verbose=0)[1],4))
    
    def evaluate_pruned(self):
        self.model.set_weights(self.current_weights_pruned)
        return(np.round(self.model.evaluate(mnist_instance.X_test, mnist_instance.y_test, verbose=0)[1],4))
    
    # Given a new weight vector a as input, updates weights of the NN accordingly
    # while keeping th structure. Updates self.current_weights_pruned weights so that 
    # percentag_pruned and pruned performance are updated
    def update_weights(self, a):
        self.current_weights = deepcopy(a)
        self.prune()    

    def prune(self):
        a = deepcopy(self.current_weights)
        nremoved = 0
        n = 0
        
        for i, ai in enumerate(a):
            # Vector case (biases)
            if len(ai.shape) == 1:
                mask = np.abs(ai) < self.threshold
                nremoved += np.sum(mask)
                ai[mask] = 0
                n += len(ai)
            
            # Matrix case (weights)
            elif len(ai.shape) == 2:
                mask = np.abs(ai) < self.threshold
                nremoved += np.sum(mask)
                ai[mask] = 0
                n += ai.size
        
        # Update pruned weights
        self.current_weights_pruned = a
        self.percentage_pruned = np.round(nremoved / n, 3)