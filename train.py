# %%writefile $script_folder/train.py

import argparse
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from azureml.core import Run 
from utils import load_data
from PIL import Image

import os, glob

WOMEN = ["hinano", "neru"]


#美女クラス
class BeautifulWomen:
    def __init__(self, data, target, target_names, images):
        self.data = data
        self.target = target
        self.target_names = target_names
        self.images = images

    #キー(インスタンス変数)を取得するメソッド
    def keys(self):
        print("[data, target, target_names, images]")
        


def load_beautiful_woman(dir):
    data = []
    target = []
    target_names = ["hinano", "neru"]
    images = []
    
    for label, woman in enumerate(WOMEN):
        file_dir = dir + woman
        files = glob.glob(file_dir + "/*.jpeg")
        print("~~~~~~~~{}の画像をNumpy形式に変換し、Listに格納中~~~~~~~~".format(woman))
        for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert('L')          #画像をグレースケールに変換
            #img = img.resize((128, 128))    #画像サイズの変更
            imgdata = np.asarray(img)       #Numpy配列に変換
            images.append(imgdata)          #画像データ: 128*128の2次元配列
            data.append(imgdata.flatten())  #画像データ: 16,384の1次元配列
            target.append(label)            #正解ラベルを格納

    print("------------ListをNumpy形式に変換中--------------")
    data = np.array(data)
    target = np.array(target)
    target_names = np.array(target_names)
    images = np.array(images)
    #インスタンスを生成
    beautifulWomen = BeautifulWomen(data, target, target_names, images)

    return beautifulWomen


test_women = load_beautiful_woman('images/test/')
train_women = load_beautiful_woman('images/train/')

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