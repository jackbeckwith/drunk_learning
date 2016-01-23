import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn import preprocessing

# create sample data set
X = np.random.random_sample([500,20])
y = np.random.randint(0,2,[500])

n,d = X.shape
nTrain = int(0.8*n)  #training on 50% of the data

# shuffle the data
idx = np.arange(n)
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# split the data
Xtrain = X[:nTrain,:]
ytrain = y[:nTrain]
Xtest = X[nTrain:,:]
ytest = y[nTrain:]

# Standardize data
scaler = preprocessing.StandardScaler().fit(X)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

# train the logistic regression
# logistic_model = LogisticRegression()
# logistic_model.fit(Xtrain, ytrain)
# y_preds_logistic = logistic_model.predict(Xtest)
# accuracy = accuracy_score(ytest, y_preds_logistic)
# print accuracy

# train naive bayes
clf = GaussianNB()
clf.fit(Xtrain, ytrain)
y_pred = clf.predict(Xtest)
accuracy = accuracy_score(ytest, y_pred)
print accuracy

filename = 'test_sklearn.pkl'
_ = joblib.dump(clf, filename, compress=9)

clf = joblib.load(filename)
print clf

# clf.partial_fit(Xtest, ytest)
y_pred = clf.predict(Xtrain)
accuracy = accuracy_score(ytrain, y_pred)
print accuracy

class DrunkLearning(object):
    """drunk_learning class"""
    def __init__(self):
        self.clf = Perceptron()
        self.filename = 'modelLogReg.pkl'

    def fit(self, X, y):
    	X = np.array([X])
        y = np.array(y)
        self.clf.fit(X, y)
        joblib.dump(self.clf, self.filename, compress=9)

    def partial_fit(self, X, y):
        X = np.array([X])
        y = np.array(y)
        self.clf.partial_fit(X, y, [0, 1])
        joblib.dump(self.clf, self.filename, compress=9)

    def predict(self, X):
    	X = np.array([X])
        ret = self.clf.predict(X)
        return str(ret[0])

class DrunkLearningSVM(DrunkLearning):
    """drunk_learning class"""
    def __init__(self):
        super(DrunkLearningSVM, self).__init__()
        self.clf = SGDClassifier(loss="hinge", penalty="l2")
        self.filename = 'modelSVM.pkl'

class DrunkLearningNB(DrunkLearning):
    """drunk_learning class"""
    def __init__(self):
        super(DrunkLearningNB, self).__init__()
        self.clf = GaussianNB()
        self.filename = 'modelNB.pkl'

class DrunkLearningNB(DrunkLearning):
    """drunk_learning class"""
    def __init__(self):
        super(DrunkLearningPassiveAggressive, self).__init__()
        self.clf = PassiveAggressiveClassifier(loss='hinge', C=1.0))
        self.filename = 'modelPassiveAggressive.pkl'

