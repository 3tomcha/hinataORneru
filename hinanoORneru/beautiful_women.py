from PIL import Image
import numpy as np
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
        
    
