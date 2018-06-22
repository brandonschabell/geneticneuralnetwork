from NeuralNetwork import NeuralNetwork
import numpy as np
import pandas as pd
import random


x = np.array(list(range(1, 10000)))
y = np.linspace(47, 4747, 9999)
z = (x + y) + random.gauss(0, 0.2)
df = pd.DataFrame({'x': x, 'y': y, 'z': z})

EPOCH_COUNT = 20

nn = NeuralNetwork(2, [2], ['linear', 'linear'], df[['x', 'y']], df[['z']], 32, EPOCH_COUNT, 1, 0.3, 0.001)
diff = nn.get_score()

print('diff={}'.format(diff))

test = pd.DataFrame({'x': [9999], 'y': [4747]})
model = nn.get_model()
model.predict(test)

def predict_value(row):
    f = row[['x', 'y']]
    f = np.expand_dims(f, axis=0)
    prediction = model.predict(f)[0][0]
    return prediction

df.loc[:,'pred'] = df.apply(lambda row: predict_value(row), axis=1)

print("Initial model summary:")
print(model.summary())

print("Mutating...")
nn.mutate()
model_mutated = nn.get_model()
print("New model summary:")
print(model_mutated.summary())