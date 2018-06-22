import NeuralNetwork
import numpy as np
import pandas as pd
import random


x = np.array(list(range(1, 1000)))
y = np.linspace(47, 4747, 999)
z = (x + y) + random.gauss(0, 0.2)
df = pd.DataFrame({'x': x, 'y': y, 'z': z})

EPOCH_COUNT = 50

#nn = NeuralNetwork.NeuralNetwork(2, [2], ['linear', 'linear'], df[['x', 'y']], df[['z']], 32, EPOCH_COUNT, 1, 0.3, 0.001)
#diff = nn.get_score()

#print("Initial model summary:")
#print(model.summary())
#
#print("Mutating...")
#nn.mutate()
#model_mutated = nn.get_model()
#print("New model summary:")
#print(model_mutated.summary())

print("Creating population")
population = NeuralNetwork.Population(count=100,
                 retain_percentage=0.7,
                 mutate_chance=0.2,
                 input_dimension=2, 
                 hidden_layer_neuron_counts=[2], 
                 activation_functions=['linear', 'linear'], 
                 x=df[['x', 'y']],
                 y=df[['z']],
                 batch_size=32,
                 epochs=EPOCH_COUNT,
                 verbose=0)

print("Evolving population...")
top_scores = []
for i in range(50):
    print('Starting round {}...'.format(i + 1))
    population.evolve()
    top_score = population.get_top_score()
    top_scores.append(top_score)
    print('Top score = {}'.format(top_score))

print("Done.")
top_nn = population.pop[0]