import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import Perceptron
import numpy as np

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df = shuffle(df)
print(df.tail()) #To see the last lines

y_train = df.iloc[0:10, 4].values
y_train = np.where(y_train == 'Iris-setosa', -1, 1)

X_train = df.iloc[0:10, :4].values

ppn = Perceptron(tol = 0.001, max_iter=10, random_state=0)
ppn.fit(X_train, y_train)

