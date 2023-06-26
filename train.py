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
    def __init__(self, imshape, categories, batchsize, lr=.0005, momentum=0):
        self.imshape, self.categories, self.bs = imshape, categories, batchsize
        w, h, d = self.imshape
        if 1:
            self.conv1 = nn.Conv2d(in_channels=self.bs, out_channels=16, kernel_size=(3,3), padding=1)
            self.lin1 = nn.Linear(in_features=16*w*h*d, out_features=64)
            self.lin2 = nn.Linear(in_features=64, out_features=self.categories)
            self.layers = [self.conv1.weight, self.lin1.weight, self.lin2.weight]
        else:
            self.lin1 = nn.Tensor.uniform(w*h*d, 256)
            self.lin2 = nn.Tensor.uniform(256, 64)
            self.lin3 = nn.Tensor.uniform(64, self.categories)
            self.layers = [self.lin1, self.lin2, self.lin3]
        self.optimizer = optim.SGD(self.layers, lr=lr, momentum=momentum)
    
    #@TinyJit
    def forward(self, X):
        '''
        #X = X.conv2d(self.conv1, padding=1).relu()
        X = X.flatten()
        X = X.dot(self.lin1).relu()
        X = X.dot(self.lin2).relu()
        X = X.dot(self.lin3).log_softmax()
        '''
        X = self.conv1(X).relu()
        X = X.flatten()
        X = self.lin1(X).relu()
        X = self.lin2(X).log_softmax()
        return X
    
    def loss(self, out, Y):
        YY = Y.flatten().astype(np.int32)
        y = np.zeros((YY.shape[0], self.categories), np.float32)
        y[range(y.shape[0]),YY] = -1.0*self.categories
        y = y.reshape(list(Y.shape)+[self.categories])
        y = nn.Tensor(y)
        return out.mul(y).mean()

    ##TinyJit
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
'''
ddir = "D:\\seiger"
data = load(ddir)
Xtrain, Ytrain, Xtest, Ytest = makedatasets(data, asnumpy=True)
'''

bs = 1
sh = Xtest[0].shape
sn = model(imshape=(sh[1], sh[0], 1), categories=10, batchsize=bs)

nn.Tensor.training = True
for i in (t := trange(3000, ncols=100, desc=f"{purple}training")):
    X, Y = sample(Xtrain, Ytrain, sn.bs)
    out, loss = sn.train(X, Y)
    loss = loss.numpy()[0]
    if i%100==0: t.set_description(f"{purple}{loss=}")
nn.Tensor.training = False

t = []
for i in trange(300, ncols=100, desc=f"{green}testing"):
    X, Y = sample(Xtest, Ytest, sn.bs)
    out = sn.forward(X)
    loss = sn.loss(out,Y)
    pred = np.argmax(out.numpy())
    t.append(pred==Y)
print(f"{cyan}the accuracy of the model is {np.mean(t)}{endc}")


while 0:
    X, Y = sample(Xtest, Ytest, 1)
    label = Y[0]
    out = sn.forward(X).numpy()
    pred = np.argmax(out)

    im = X.reshape(sn.imshape[0], sn.imshape[1]).numpy()
    cv2.imshow("im", im)
    print(f"{purple}{label=}, \n{green[:]}{out=}, \n{cyan}{pred}{endc}\n")
    cv2.waitKey(0)

















