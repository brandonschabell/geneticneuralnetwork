import GeneticNeuralNetwork
import numpy as np
import pandas as pd
import random


x = np.array(list(range(0, 1000)))
y = np.linspace(47, 4747, 1000)
z = (x + y) + random.gauss(0, 0.2)
df = pd.DataFrame({'x': x, 'y': y, 'z': z})

#print("Creating complex population")
#population = GeneticNeuralNetwork.Population(count=50,
#                                             x=df[['x', 'y']],
#                                             y=df[['z']],
#                                             retain_percentage=0.6,
#                                             mutate_chance=0.3,
#                                             max_epoch=50,
#                                             min_epoch=1,
#                                             min_lr=0.0001,
#                                             max_lr=0.1,
#                                             percentage_to_randomly_spawn=0.05,
#                                             batch_size=256,
#                                             verbose=0)

print("Creating simple population")
population = GeneticNeuralNetwork.Population(count=25,
                                             x=df[['x', 'y']],
                                             y=df[['z']])

print("Evolving population...")
top_scores = []
for i in range(3):
    print('Generation {}:'.format(i + 1))
    population.evolve()
    top_score = population.get_top_score()
    top_scores.append(top_score)
    print('Top score = {}'.format(top_score))

print("Done.")

def show_model_info(index):
    population.population[index].show_info()

print("Top model info:")
show_model_info(0)
