import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import random


class NeuralNetwork:
    mutatable_params = ['neuron_counts', 'activation_functions', 'layer_count']
    
    def r2_keras(y_true, y_pred):
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )


    def diff(y_true, y_pred):
        return K.mean(abs(y_pred - y_true))
    
    
    def __init__(self, 
                 input_dimension, 
                 hidden_layer_neuron_counts, 
                 activation_functions, 
                 x=None,
                 y=None,
                 batch_size=None,
                 epochs=1,
                 verbose=1,
                 validation_split=0., 
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

