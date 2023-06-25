import os, cv2, random, numpy as np
from tinygrad import nn


def imexists(p):
    return os.path.isfile(p)

def sample(Xtrain, Ytrain, num):
    samp = np.random.randint(0, Xtrain.shape[0], size=(num))
    xx = Xtrain[samp]
    sh = xx.shape
    X = nn.Tensor(xx, requires_grad=False).reshape(1, num, sh[1], sh[2])
    Y = Ytrain[samp]
    return X, Y

def makedataset(data, asnumpy=False):
    train = data[0:len(data)*9//10]
    test = data[len(data)*9//10:-1]

    Xtrain, Ytrain, = [], []
    Xtest, Ytest = [], []
    for i, (xt, yt) in enumerate(train):
        Xtrain.append(xt)
        Ytrain.append(yt)
    for i, (xt, yt) in enumerate(test):
        Xtest.append(xt)
        Ytest.append(yt)
    if asnumpy: Xtrain, Ytrain, Xtest, Ytest = np.array(Xtrain), np.array(Ytrain), np.array(Xtest), np.array(Ytest)
    else: Xtrain, Ytrain, Xtest, Ytest = nn.Tensor(Xtrain), nn.Tensor(Ytrain), nn.Tensor(Xtest), nn.Tensor(Ytest)
    return Xtrain, Ytrain, Xtest, Ytest

def load(imgdir, num=440):
    data = []
    for i in range(6):
        ims, labels = getimgs(f"{imgdir}\\{i}", num, label=i)
        for z in range(len(ims)):
            im, label = ims[z], int(labels[z])
            im = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)/255).astype(np.float32)
            #hot = np.zeros(6, dtype=np.float32)
            #hot[label] = 1
            #data.append((nn.Tensor(im), nn.Tensor(hot)))
            data.append((im, label))
    random.shuffle(data)
    return data

def getimgs(dir, num, label=None, min=0, filetype="png"):
    ims, labels = [], []
    impath = f"{dir}\\{min}.{filetype}"
    exists = imexists(impath) 
    assert exists, f"attempt to load image  at {impath} failed!"
    if num==-1:
        i = min
        while exists:
            im = cv2.imread(impath)
            ims.append(im)
            labels.append(label)
            i += 1
            impath = f"{dir}\\{i}.{filetype}"
            exists = imexists(impath) 
    elif num > 0:
        for i in range(min, num):
            impath = f"{dir}\\{i}.{filetype}"
            exists = imexists(impath) 
            assert exists, f"attempt to load image  at {impath} failed!"
            im = cv2.imread(impath)
            ims.append(im)
            labels.append(label)
    if label==None: return ims
    return ims, labels

purple = '\033[95m'
blue = '\033[94m'
cyan = '\033[96m'
green = '\033[92m'
yellow = '\033[93m'
red = '\033[91m'
endc = '\033[0m'
bold = '\033[1m'
underline = '\033[4m'