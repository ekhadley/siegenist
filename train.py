import cv2, random, time, numpy as np
from tqdm import trange
from tinygrad import nn
from tinygrad.nn import optim
from tinygrad.jit import TinyJit
from tinygrad.lazy import Device
from funcs import *
from mnist import MNIST
Device.default = "GPU"

class model:
    def __init__(self, imshape, categories, lr=.01):
        self.imshape, self.categories = imshape, categories
        w, h, d = self.imshape
        self.conv1 = nn.Tensor.uniform(16, 1, 3, 3)
        self.lin1 = nn.Tensor.uniform(16*w*h, 128)
        self.lin2 = nn.Tensor.uniform(128, self.categories)
        self.layers = [self.conv1, self.lin1, self.lin2]
        #self.optimizer = optim.SGD(self.layers, lr=lr, momentum=momentum)
        self.optimizer = optim.AdamW(self.layers, lr=lr)
    
    #@TinyJit
    def forward(self, X):
        X = X.reshape(-1, 1, self.imshape[0], self.imshape[1])
        X = X.conv2d(self.conv1, padding=1)
        X = X.reshape(-1, self.imshape[0]*self.imshape[1]*self.conv1.shape[0])
        #print(f"{red}{X.shape=}{endc}")
        X = X.dot(self.lin1).relu()
        X = X.dot(self.lin2).log_softmax()
        return X
    
    def loss(self, out, Y):
        YY = Y.flatten().astype(np.int32)
        y = np.zeros((YY.shape[0], self.categories), np.float32)
        y[range(y.shape[0]),YY] = -1.0*self.categories
        y = y.reshape(list(Y.shape)+[self.categories])
        y = nn.Tensor(y)
        return out.mul(y).mean()

    #@TinyJit
    def train(self, X, Y):
        out = self.forward(X)
        loss = self.loss(out,Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return out, loss

ddir = "D:\\wgmn\\tinygrad\\datasets\\mnist"
mn = MNIST(path=ddir, return_type="numpy")
Xtrain, Ytrain = mn.load_training()
Xtrain, Ytrain = np.reshape(Xtrain, (60_000, 28, 28)).astype(np.float32), Ytrain.astype(np.float32)
Xtest, Ytest = mn.load_testing()
Xtest, Ytest = np.reshape(Xtest, (10_000, 28, 28)).astype(np.float32), Ytest.astype(np.float32)
Xtest = Xtest/255.0
Xtrain = Xtrain/255.0
'''
ddir = "D:\\seiger"
data = load(ddir)
Xtrain, Ytrain, Xtest, Ytest = makedatasets(data, asnumpy=True)
Xtrain = Xtrain.reshape((-1, 20, 30))
Xtest = Xtest.reshape((-1, 20, 30))
'''

bs = 32
sh = Xtest[0].shape
sn = model(imshape=(sh[1], sh[0], 1), categories=10)

nn.Tensor.training = True
for i in (t := trange(2000, ncols=100, desc=f"{purple}training")):
    X, Y = sample(Xtrain, Ytrain, bs)
    out, loss = sn.train(X, Y)
    loss = loss.numpy()[0]
    if i%10==0:
        #print(f"\n{cyan}{out.numpy()=}{endc}")
        t.set_description(f"{purple}{loss=}")

t = []
for i in trange(300, ncols=100, desc=f"{green}testing"):
    X, Y = sample(Xtest, Ytest, 1)
    out = sn.forward(X)
    loss = sn.loss(out,Y)
    pred = np.argmax(out.numpy())
    t.append(pred==Y)
print(f"{cyan}the accuracy of the model is {np.mean(t)}{endc}")


while 1:
    X, Y = sample(Xtest, Ytest, 1)
    label = Y[0]
    out = sn.forward(X).numpy()
    pred = np.argmax(out)

    im = X.reshape(sn.imshape[0], sn.imshape[1]).numpy()
    cv2.imshow("im", im)
    print(f"{purple}{label=}, \n{green[:]}{out=}, \n{cyan}{pred}{endc}\n")
    cv2.waitKey(0)

















