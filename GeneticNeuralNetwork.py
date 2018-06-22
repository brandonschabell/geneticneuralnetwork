from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.optimizers import Adam
import random
import numpy as np
import math
from copy import deepcopy
import sys

ACTIVATIONS_LIST = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 
                    'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
MUTATABLE_PARAMS = ['neuron_counts', 'activation_functions', 'layer_count', 'learning_rate', 'epoch_num']

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
                 max_epoch=200,
                 min_epoch=1,
                 min_lr=0.0001,
                 max_lr=0.1,
                 percentage_to_randomly_spawn=0.05,
                 x=None,
                 y=None,
                 mutate_chance=0.2,
                 batch_size=None,
                 verbose=1,
                 validation_split=0.3, 
                 learning_rate=0.001, 
                 kernel_initializer='normal', 
                 loss_function='mean_squared_error', 
                 metrics=[r2_keras, diff]):
        self.population = []
        self.count = count
        self.input_dimension = input_dimension
        self.percentage_to_randomly_spawn = percentage_to_randomly_spawn
        self.x = x
        self.y = y
        self.retain_percentage = retain_percentage
        self.mutate_chance = mutate_chance
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.kernel_initializer = kernel_initializer
        self.loss_function = loss_function
        self.metrics = metrics
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        for i in range(count):
            self.population.append(self.create_random_net())
    
    
    def evolve(self):
        net_count = len(self.population)
        net_iter = 0
        graded = []
        for nn in self.population:
            net_iter += 1
            sys.stdout.flush()
            sys.stdout.write("\rChecking net #{} of {}".format(net_iter, net_count))
            graded.append((nn.get_score(), nn))
        print()
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=False)]
        retained_length = int(len(graded) * self.retain_percentage)
        keep = graded[:retained_length]
        m_count = 0
        s_count = 0
        b_count = 0
        for nn in keep:
            if self.mutate_chance > random.random():
                new_nn = deepcopy(nn)
                m_count += 1
                keep.append(new_nn.mutate())
        print("Mutated {} nets.".format(m_count))
        
        for i in range(int(self.count * self.percentage_to_randomly_spawn)):
            keep.append(self.create_random_net())
            s_count += 1
        print("Spawned {} nets.".format(s_count))
            
        while len(keep) < self.count:
            net1 = random.randint(0, retained_length - 1)
            net2 = random.randint(0, retained_length - 1)
            if net1 != net2:
                b_count += 1
                keep.append(keep[net1].breed(keep[net2]))
        print("Bred {} nets.".format(b_count))
        self.population = keep

    
    def get_top_score(self):
        return self.population[0].get_score()
    
    
    def create_random_net(self):
        hlnc = []
        acts = [random.choice(ACTIVATIONS_LIST)]
        layer_count = random.randint(0, 10)
        for l in range(layer_count):
            hlnc.append(random.randint(
                        math.ceil(self.input_dimension * 0.5),
                        self.input_dimension * 2))
            acts.append(random.choice(ACTIVATIONS_LIST))
        return NeuralNetwork(input_dimension=self.input_dimension, 
                           hidden_layer_neuron_counts=hlnc, 
                           activation_functions=acts,
                           min_lr=self.min_lr,
                           max_lr=self.max_lr,
                           min_epoch=self.min_epoch,
                           max_epoch=self.max_epoch,
                           x=self.x,
                           y=self.y,
                           batch_size=self.batch_size,
                           epochs=None,
                           verbose=self.verbose,
                           validation_split=self.validation_split, 
                           learning_rate=None, 
                           kernel_initializer=self.kernel_initializer, 
                           loss_function=self.loss_function, 
                           metrics=self.metrics)
    

class NeuralNetwork:
    def __init__(self, 
                 input_dimension, 
                 hidden_layer_neuron_counts, 
                 activation_functions,
                 min_lr,
                 max_lr,
                 min_epoch,
                 max_epoch,
                 x=None,
                 y=None,
                 batch_size=None,
                 epochs=None,
                 verbose=1,
                 validation_split=0.3, 
                 learning_rate=None, 
                 kernel_initializer='normal', 
                 loss_function='mean_squared_error', 
                 metrics=[r2_keras, diff]):
        self.input_dimension = input_dimension
        self.hidden_layer_neuron_counts = hidden_layer_neuron_counts
        self.activation_functions = activation_functions
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.x = x
        self.y = y
        self.batch_size = batch_size
        if epochs is None:
            self.epochs = self.new_epoch_num()
        else:
            self.epochs = epochs
        self.verbose = verbose
        self.validation_split = max(validation_split, 0.1)
        if learning_rate is None:
            self.learning_rate = self.new_learning_rate()
        else:
            self.learning_rate = learning_rate
        self.kernel_initializer = kernel_initializer
        self.loss_function = loss_function
        self.metrics = metrics
        self.model = Sequential()
        self.score = None
        self.is_processed = False
    
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.input_dimension = self.input_dimension
        result.hidden_layer_neuron_counts = self.hidden_layer_neuron_counts
        result.activation_functions = self.activation_functions
        result.min_lr = self.min_lr
        result.max_lr = self.max_lr
        result.min_epoch = self.min_epoch
        result.max_epoch = self.max_epoch
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
        result.score = None
        result.is_processed = False
        return result
    
    
    def mutate(self):
        self.model = Sequential()
        self.is_processed = False
        self.score = None
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
        elif param == 'learning_rate':
            self.learning_rate = self.new_learning_rate(self.learning_rate)
        elif param == 'epoch_num':
            self.epochs = self.new_epoch_num(self.epochs)
        return self
    
    
    def breed(self, mate):
        child = deepcopy(self)
        new_hlnc = []
        new_acts = []
        if len(self.hidden_layer_neuron_counts) == len(mate.hidden_layer_neuron_counts):
            for c in range(len(self.hidden_layer_neuron_counts)):
                if random.random() > 0.5:
                    new_hlnc.append(self.hidden_layer_neuron_counts[c])
                else:
                    new_hlnc.append(mate.hidden_layer_neuron_counts[c])
            child.hidden_layer_neuron_counts = new_hlnc
        if len(self.activation_functions) == len(mate.activation_functions):
            for c in range(len(self.activation_functions)):
                if random.random() > 0.5:
                    new_acts.append(self.activation_functions[c])
                else:
                    new_acts.append(mate.activation_functions[c])
            child.activation_functions = new_acts
        child.epochs = int(np.array([self.epochs, mate.epochs]).mean())
        child.learning_rate = np.array([self.learning_rate, mate.learning_rate]).mean()
        return child
    
    
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
        if self.score is None:
            history = model.fit(self.x, 
                                self.y, 
                                batch_size=self.batch_size, 
                                epochs=self.epochs, 
                                verbose=self.verbose,
                                validation_split=self.validation_split).history
            self.score = history['val_diff'][-1]
        return self.score
    
    
    def new_learning_rate(self, cur_val=None):
        if cur_val is None:
            return random.uniform(self.min_lr, self.max_lr)
        else:
            new_val = np.random.normal(cur_val, cur_val)
            if new_val >= self.min_lr and new_val <= self.max_lr:
                return new_val
            else:
                return self.new_learning_rate(cur_val)


    def new_epoch_num(self, cur_val=None):
        if cur_val is None:
            return random.randint(self.min_epoch, self.max_epoch)
        else:
            new_val = int(np.random.normal(cur_val, min(cur_val, 10)))
            if new_val >= self.min_epoch and new_val != cur_val and new_val <= self.max_epoch:
                return new_val
            else:
                return self.new_epoch_num(cur_val)
    
    
    def show_info(self):
        print(self.get_model().summary())
        print('Epochs = {}'.format(self.epochs))
        print('Learning rate = {}'.format(self.learning_rate))
        print('Activation functions = {}'.format(self.activation_functions))

