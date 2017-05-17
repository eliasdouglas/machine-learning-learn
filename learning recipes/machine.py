# import a dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .5)

# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, Y_train)

predictions = my_classifier.predict(X_test)
# print(predictions)
# list of numbers.. thease correspond to the type of Iris
# the classifier predicts for each row in the testing data.

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))


# replacing the comment line 11 and 12 for 14 and 15.