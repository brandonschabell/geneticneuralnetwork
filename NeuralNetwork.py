from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.optimizers import Adam
import random
import math
from copy import deepcopy

ACTIVATIONS_LIST = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 
                    'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
MUTATABLE_PARAMS = ['neuron_counts', 'activation_functions', 'layer_count']

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def diff(y_true, y_pred):
    return K.mean(abs(y_pred - y_true))


class Population:
    def __init__(self, 
                 count,
                 retain_percentage,
                 input_dimension, 
                 hidden_layer_neuron_counts, 
                 activation_functions, 
                 x=None,
                 y=None,
                 mutate_chance=0.2,
                 batch_size=None,
                 epochs=1,
                 verbose=1,
                 validation_split=0.3, 
                 learning_rate=0.001, 
                 kernel_initializer='normal', 
                 loss_function='mean_squared_error', 
                 metrics=[r2_keras, diff]):
        self.pop = []
        self.retain_percentage = retain_percentage
        self.mutate_chance = mutate_chance
        for i in range(count):
            nn = NeuralNetwork(input_dimension,
                               hidden_layer_neuron_counts, 
                               activation_functions, 
                               x,
                               y,
                               batch_size,
                               epochs,
                               verbose,
                               validation_split, 
                               learning_rate,
                               kernel_initializer, 
                               loss_function, 
                               metrics)
            if i > 0:
                while random.random() > 0.2:
                    nn.mutate()
            self.pop.append(nn)
    
    
    def evolve(self):
        graded = [(nn.get_score(), nn) for nn in self.pop]
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=False)]
        retained_length = int(len(graded) * self.retain_percentage)
        keep = graded[:retained_length]
        for nn in keep:
            if self.mutate_chance > random.random():
                new_nn = deepcopy(nn)
                keep.append(new_nn.mutate())
            
        while len(keep) < len(self.pop):
            new_nn = deepcopy(keep[0])
            new_nn.mutate()
            while random.random() > 0.4:
                new_nn.mutate()
            keep.append(new_nn)
        self.pop = keep

    
    def get_top_score(self):
        return self.pop[0].get_score()
    

class NeuralNetwork:
    def __init__(self, 
                 input_dimension, 
                 hidden_layer_neuron_counts, 
                 activation_functions, 
                 x=None,
                 y=None,
                 batch_size=None,
                 epochs=1,
                 verbose=1,
                 validation_split=0.3, 
                 learning_rate=0.001, 
                 kernel_initializer='normal', 
                 loss_function='mean_squared_error', 
                 metrics=[r2_keras, diff]):
        self.input_dimension = input_dimension
        self.hidden_layer_neuron_counts = hidden_layer_neuron_counts
        self.activation_functions = activation_functions
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.validation_split = max(validation_split, 0.1)
        self.learning_rate = learning_rate
        self.kernel_initializer = kernel_initializer
        self.loss_function = loss_function
        self.metrics = metrics
        self.model = Sequential()
        self.is_processed = False
    
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.input_dimension = self.input_dimension
        result.hidden_layer_neuron_counts = self.hidden_layer_neuron_counts
        result.activation_functions = self.activation_functions
        result.x = self.x
        result.y = self.y
        result.batch_size = self.batch_size
        result.epochs = self.epochs
        result.verbose = self.verbose
        result.validation_split = self.validation_split
        result.learning_rate = self.learning_rate
        result.kernel_initializer = self.kernel_initializer
        result.loss_function = self.loss_function
        result.metrics = self.metrics
        result.model = Sequential()
        result.is_processed = False
        return result
    
    
    def mutate(self):
        self.model = Sequential()
        self.is_processed = False
        param = random.choice(MUTATABLE_PARAMS)
        if param == 'neuron_counts':
            if len(self.hidden_layer_neuron_counts) == 0:
                return self.mutate()
            index = random.randint(0, len(self.hidden_layer_neuron_counts) - 1)
            cur_val = self.hidden_layer_neuron_counts[index]
            if random.random() > 0.5:
                self.hidden_layer_neuron_counts[index] = cur_val + 1
            else:
                if cur_val == 1:
                    return self.mutate()
                self.hidden_layer_neuron_counts[index] = cur_val - 1
        elif param == 'activation_functions':
            index = random.randint(0, len(ACTIVATIONS_LIST) - 1)
            new_val = random.choice(ACTIVATIONS_LIST)
            ACTIVATIONS_LIST[index] = new_val
        elif param == 'layer_count':
            if random.random() > 0.5:
                new_neuron_count = random.randint(
                        math.ceil(self.input_dimension * 0.5),
                        self.input_dimension * 2)
                self.hidden_layer_neuron_counts.insert(0, new_neuron_count)
                self.activation_functions.insert(1, ACTIVATIONS_LIST[0])
            elif len(self.hidden_layer_neuron_counts) > 0:
                index = random.randint(0, len(self.hidden_layer_neuron_counts) - 1)
                del self.hidden_layer_neuron_counts[index]
                del self.activation_functions[index + 1]
            else:
                return self.mutate()
        return self
    
    
    def process_model(self):
        if len(self.hidden_layer_neuron_counts) > 0:
            first_output_count = self.hidden_layer_neuron_counts[0]
        else:
            first_output_count = 1
        self.model.add(Dense(input_dim=self.input_dimension,
                        activation=self.activation_functions[0],
                        units=first_output_count,
                        kernel_initializer=self.kernel_initializer))
        for i in range(1, len(self.hidden_layer_neuron_counts)):
            self.model.add(Dense(units=self.hidden_layer_neuron_counts[i], activation=self.activation_functions[i]))
        if len(self.hidden_layer_neuron_counts) > 0:
            self.model.add(Dense(units=1, activation=self.activation_functions[-1]))
        optimizer = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.loss_function, metrics=self.metrics)
    
    
    def get_model(self):
        if (not self.is_processed):
            self.process_model()
            self.is_processed = True
        return self.model
    
    
    def get_score(self):
        model = self.get_model()
        history = model.fit(self.x, 
                            self.y, 
                            batch_size=self.batch_size, 
                            epochs=self.epochs, 
                            verbose=self.verbose,
                            validation_split=self.validation_split).history
        diff = history['val_diff']
        return diff[-1]

