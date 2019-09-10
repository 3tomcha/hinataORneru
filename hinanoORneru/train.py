
import argparse
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from azureml.core import Run 
from utils import load_data

import beautiful_women

test_women = beautiful_women.load_beautiful_woman('../images/test/')
train_women = beautiful_women.load_beautiful_woman('../images/train/')

parser = argparse.ArgumentParser()
# parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
# parser.add_argument('--test', type=BeautifulWomen, dest='test')
# parser.add_argument('--train', type=BeautifulWomen, dest='train')
parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regulation rate')
args = parser.parse_args()

X_train = train_women.data
X_test = test_women.data
y_train = train_women.target
y_test = train_women.target

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep = '\n')

run = Run.get_context()
print('Train a logistic regression model with regularization rate of', args.reg)
clf = LogisticRegression(C=1.0/args.reg, solver="liblinear", multi_class="auto", random_state=42)
clf.fit(X_train, y_train)

print('Predict the test set')
y_hat = clf.predict(X_test)

acc = np.average(y_hat == y_test)
print('Accuracy is ', acc)

run.log('regularization rate', np.float(args.reg))
run.log('accuracy', np.float(acc))

os.makedirs('outputs', exist_ok=True)
joblib.dump(value=clf, filename='outputs/sklearn_mnist_model.pkl')