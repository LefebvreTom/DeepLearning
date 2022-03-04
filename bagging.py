from tkinter.tix import Tree
import pandas as pd
#import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.utils import shuffle
#from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
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

tree = DecisionTreeClassifier(criterion = 'entropy', max_depth= 20, random_state=1)
bag = BaggingClassifier(base_estimator= tree, n_estimators=500, max_samples= 1.0, max_features= 1.0, bootstrap = True, bootstrap_features = False, n_jobs = 1, random_state=1)

tree.fit(X_train_std, y_train)
print("tree : ")
print(tree.score(X_test_std, y_test))
bag.fit(X_train_std, y_train)
print("bagging : ")
print(bag.score(X_test_std, y_test))