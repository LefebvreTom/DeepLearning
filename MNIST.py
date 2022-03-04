import os
import sys
import struct
import numpy as np
import matplotlib as plt

from sklearn.neural_network import MLPClassifier
import MLP as mlp
 
#Les images sont dans le format Byte, on les mets dans un NP
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte' % kind)
	
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8)) #Utiliser pour la lecture du format, > veut dire Big Endian et I entier positif
        labels = np.fromfile(lbpath,dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16)) #m�me chose
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
		
    return images, labels
	
	
#On retourne deux matrices
#Images est la matrice n x m avec n=nombre d'�chantillons et m=nombre de features
#labels contient m entiers de 0-9 correspondants aux classes


X_train, y_train = load_mnist(r'C:/Users/reyna/Desktop/Travail/UQAC/Trimestre2-Hiver/DeepLearning/Travail4/mnist/', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

#Rows: 60000, columns: 784

X_test, y_test = load_mnist(r'C:/Users/reyna/Desktop/Travail/UQAC/Trimestre2-Hiver/DeepLearning/Travail4/mnist/', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

#Rows: 10000, columns: 784

nn = mlp.NeuralNetMLP(n_output=10, 
                  n_features=X_train.shape[1], 
                  n_hidden=60, 
                  l2=0.15, 
                  l1=0.0, 
                  epochs=1000, 
                  eta=0.0001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  minibatches=50, 
                  shuffle=True,
                  random_state=1)


nn.fit(X_train, y_train, print_progress=True)

import matplotlib.pyplot as plt

plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
# plt.savefig('./figures/cost.png', dpi=300)
plt.show()


#y_train_pred = nn.predict(X_train)

#if sys.version_info < (3, 0):
#    acc = ((np.sum(y_train == y_train_pred, axis=0)).astype('float') /
#           X_train.shape[0])
#else:
#    acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]

#print('Training accuracy: %.2f%%' % (acc * 100))


y_test_pred = nn.predict(X_test)

if sys.version_info < (3, 0):
    acc = ((np.sum(y_test == y_test_pred, axis=0)).astype('float') /
           X_test.shape[0])
else:
    acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]

print('Test accuracy MLP custom: %.2f%%' % (acc * 100))



mlpscikit = MLPClassifier(hidden_layer_sizes=60, alpha= 0.001, learning_rate_init= 0.0001, random_state= 1, shuffle=True)
mlpscikit.fit(X_train, y_train)
print("Score MLP scikit learn :")
print(mlpscikit.score(X_test, y_test))