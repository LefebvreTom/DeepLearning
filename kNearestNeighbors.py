import pandas as pd
#import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.utils import shuffle
#from mlxtend.plotting import plot_decision_regions
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = arff.loadarff(r'C:\Users\reyna\Desktop\Travail\UQAC\Trimestre2-Hiver\DeepLearning\Travail2\RFID_Features_windows5.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8') 
df = shuffle(df)
print(df.tail()) #To see the last lines

X = df.iloc[:, :188].values
y = df.iloc[:, 188].values
X_train, X_test, y_train, y_test = train_test_split(X ,y ,test_size = 0.3 , random_state = 1)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#y_train = np.where(y_train == 'Chambre_a1', 1, -1)
#y_test = np.where(y_test == 'Chambre_a1', 1, -1)


knn = KNeighborsClassifier(n_neighbors= 5, p= 2, metric= 'minkowski')
knn.fit(X_train_std, y_train)
print("k nearest neighbors : ")
print(knn.score(X_test_std, y_test))