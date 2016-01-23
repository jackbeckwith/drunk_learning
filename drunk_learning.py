import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# train the logistic regression
logistic_model = LogisticRegression()
logistic_model.fit(Xtrain, ytrain)
y_preds_logistic = logistic_model.predict(Xtest)
accuracy = accuracy_score(ytest, y_preds_logistic)
print accuracy
