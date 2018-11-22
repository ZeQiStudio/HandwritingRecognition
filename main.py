import numpy as np
import os
import struct
import random
def softmax(X):
    X=X-np.max(X)
    expX=np.exp(X)
    softmaxX=expX/np.sum(expX)
    return softmaxX


def Normalization(X):
    X=X-np.average(X)
    X=X/X.max()
    return X
def LoadData(path,kind):
    labelPath=os.path.join(path,'%s-labels.idx1-ubyte'%kind)
    imagePath=os.path.join(path,'%s-images.idx3-ubyte'%kind)
    with open(labelPath,'rb') as lb:
        magic,n=struct.unpack('>II',lb.read(8))
        labels=np.fromfile(lb,dtype=np.uint8)
    with open(imagePath,'rb') as ib:
        magic, num, rows, cols = struct.unpack('>IIII',ib.read(16))
        images = np.fromfile(ib,dtype=np.uint8).reshape(len(labels), 784)
    labelsResult=[]
    for i in range(labels.shape[0]):
        y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        y[labels[i]]=1
        labelsResult.append(y)
    labelsResult=np.array(labelsResult)
    images=Normalization(images)
    return images,labelsResult

def RandomInitial(x,y):
    w=np.random.rand(x,y)
    b=random.random()
    b=b*np.ones((1,y))
    return w,b

def Learning(X_train,y_train,learningRate,learningTimes,mylambda):
    w1,b1=RandomInitial(784,50)
    w2,b2=RandomInitial(50,10)
    num=y_train.shape[0]
    for times in range(learningTimes):
        DW1 = np.zeros((784, 50))
        DW2 = np.zeros((50, 10))
        DB1 = np.zeros((1, 50))
        DB2 = np.zeros((1, 10))
        cost = 0
        for i in range(num):
            X=X_train[i].reshape(1,784)
            y=y_train[i]

            #FB
            z2=np.dot(X,w1)+b1
            a2=np.tanh(z2)

            z3=np.dot(a2,w2)+b2
            h=softmax(z3)

            #BP
            delta3=h-y
            DW2=np.dot(a2.T,delta3)/num+mylambda*w2
            DB2=delta3.sum()/num
            delta2=np.multiply(np.dot(delta3,w2.T),np.power(a2,2))
            DW1=np.dot(X.T,delta2)/num+mylambda*w1
            DB1=delta2.sum()/num
            #标准梯度下降
            w2=w2-learningRate*DW2
            w1=w1-learningRate*DW1
            b1=b1-learningRate*DB1
            b2=b2-learningRate*DB2
            #Cost
            J=np.multiply(-y,np.log(h)).sum()/num
            cost=cost+J
        print('Cost:',times+1,cost)
        #累积梯度下降
        #w2=w2-learningRate*DW2
        #w1=w1-learningRate*DW1
        #b1=b1-learningRate*DB1
        #b2=b2-learningRate*DB2
    return w1,w2,b1,b2
def HProcess(X):
    max=X[0][0]
    index=0
    for i in range(X.shape[1]):
        if max<X[0][i]:
            max=X[0][i]
            index=i
        X[0][i]=0
    X[0][index]=1

def Predict(X_test,y_test,w1,w2,b1,b2):
    num=y_test.shape[0]
    corrAns=0
    for i in range(num):
        X = X_test[i].reshape(1, 784)
        y = y_test[i]
        # FP
        z2 = np.dot(X, w1) + b1
        a2 = np.tanh(z2)
        z3 = np.dot(a2, w2) + b2
        h = softmax(z3)

        index=HProcess(h)
        if((h==y).all()==True):
            corrAns=corrAns+1
    print('Correct/Total:',corrAns,'/',num)
    print('CorrectRate:',corrAns/num*100,'%')






#-----------------------------------------
#ReadData
X_train,y_train=LoadData(os.getcwd(),'train')
learningRate=0.4
learningTimes=10000
w1,w2,b1,b2=Learning(X_train,y_train,learningRate,learningTimes,0)
X_test,y_test=LoadData(os.getcwd(),'t10k')
Predict(X_test,y_test,w1,w2,b1,b2)
