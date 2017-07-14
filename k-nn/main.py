# Very simple KNN implementation on the iris dataset

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from knn import KNN

# Importing the dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

classifier = KNN()
classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test)

print(accuracy_score(Y_test, predictions))

