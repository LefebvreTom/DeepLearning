import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import Perceptron
import numpy as np

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)
df = shuffle(df)
print(df.tail()) #To see the last lines

y_train = df.iloc[0:10, 9].values
y_train = np.where(y_train == 'positive', -1, 1)

X_train = df.iloc[0:10, :9].values
X_train2 = []
for x in X_train:
    x2 = []
    for xi in x:
        if xi == 'o':
            x2.append(np.float64(0))
        elif xi == 'b':
            x2.append(np.float64(1))
        elif xi == 'x':
            x2.append(np.float64(2))
    X_train2.append(x2)
X_train2 = np.asarray(X_train2)

ppn = Perceptron(tol = 0.001, max_iter=15, random_state=0)
ppn.fit(X_train2, y_train)

