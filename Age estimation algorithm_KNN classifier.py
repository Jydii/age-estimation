import numpy as np # linear algebra
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, confusion_matrix, classification_report, roc_auc_score, auc
import pickle
from sklearn.decomposition import PCA

import time
start_time = time.time()


label_num = 0

#load data
with open('93ratios.pickle', 'rb') as f:
  data = pickle.load(f)

i = 0
x = np.empty( (len(data), 93)) #numpy size = num_of_person, (img_size*img_size+(x,y,w,h)*num_of_parts)
y = np.empty(len(data))

for person in data:
	item = np.concatenate((
		person['1'], person['2'], person['3'], person['4'], person['5'], person['6'], person['7'], person['8'], person['9'], person['10'],
		person['11'], person['12'], person['13'], person['14'], person['15'], person['16'], person['17'], person['18'], person['19'], person['20'],
		person['21'], person['22'], person['23'], person['24'], person['25'], person['26'], person['27'], person['28'], person['29'], person['30'],
		person['31'], person['32'], person['33'], person['34'], person['35'], person['36'], person['37'], person['38'], person['39'], person['40'],
		person['41'], person['42'], person['43'], person['44'], person['45'], person['46'], person['47'], person['48'], person['49'], person['50'],
		person['51'], person['52'], person['53'], person['54'], person['55'], person['56'], person['57'], person['58'], person['59'], person['60'],
		person['61'], person['62'], person['63'], person['64'], person['65'], person['66'], person['67'], person['68'], person['69'], person['70'],
		person['71'], person['72'], person['73'], person['74'], person['75'], person['76'], person['77'], person['78'], person['79'], person['80'],
		person['81'], person['82'], person['83'], person['84'], person['85'], person['86'], person['87'], person['88'], person['89'], person['90'],
		person['91'], person['92'], person['93']
	), axis=None )



	x[i] = item

	if person['label'] == 'Child':
		label_num = 0
	else:
		label_num = 1

	y[i] = label_num
	i = i + 1





x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=42)


# Compute a PCA
n_components = 21
pca = PCA(n_components=n_components, whiten=True).fit(x_train)

# apply PCA transformation
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)


#Setup arrays to store training and test accuracies
neighbors = np.arange(1,10)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
	# Setup a knn classifier with k neighbors
	knn = KNeighborsClassifier(n_neighbors=k)

	# Fit the model
	knn.fit(x_train_pca, y_train)

	# Compute accuracy on the training set
	train_accuracy[i] = knn.score(x_train_pca, y_train)

	# Compute accuracy on the test set
	test_accuracy[i] = knn.score(x_test_pca, y_test)

#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=5)

print('Start training...')

#Fit the model
knn.fit(x_train_pca, y_train)
print('Start testing...')

y_pred = knn.predict(x_test_pca)
acc = knn.score(x_test_pca, y_test)
train_acc = knn.score(x_train_pca, y_train)

#print(classification_report(y_test, y_pred))

print('KNN Test accuracy:')
print(acc)
print('KNN Train accuracy:')
print(train_acc)

mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:')
print(mae)

print(classification_report(y_test, y_pred))
# IMPORTANT: first argument is true values, second argument is predicted values
# this produces a 2x2 numpy array (matrix)
conf_mtx = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_mtx)

print('Execution Time:')
print("%s seconds" % (time.time() - start_time))



