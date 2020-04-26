# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:08:05 2020

@author: arutema47
"""

import numpy as np

class Variable():
    def __init__(self, x):
        # Input must be np.array
        if x is not None:
            if not isinstance(x, np.ndarray):
                raise TypeError("{} is not supported".format(type(x)))
                
        self.data = x
        self.grad = None
        self.creator = None
    
    # for define-by-run
    def set_creator(self, func):
        self.creator = func
        
    # 微分をリセットする
    def cleargrad(self):
        self.grad = None
        
    # for auto-grad
    def backward(self):
        # for initial grad
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        # define-by-run in a loop fasion
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            # 複数grad情報を読み込み
            gys = [output.grad for output in f.outputs]
            # gradをunpackして逆伝搬を計算
            gxs = f.backward(*gys)
            # tuple化
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            
            # gradを変数xに格納
            for x, gx in zip(f.inputs, gxs):
                # 同じ変数を複数回使用したときの対策
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                # 一つ前の関数を読みに行く。
                if x.creator is not None:
                    funcs.append(x.creator)
 
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
       
class Function ():
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        # unpack inputs and pass to forward
        ys = self.forward(*xs)
        # make output tuple if it is not tuple.
        if not isinstance(ys, tuple):
            ys = (ys, )
            
        # set output for backpropagation
        outputs = [Variable(as_array(y)) for y in ys]
        # set the creator of the output, which is the function itself.
        for output in outputs:
            output.set_creator(self)
        # remember input, outputs
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()
        
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx
    
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0,x1):
    return Add()(x0,x1)

if __name__ == "__main__":    
    print("test of simple addition..")
    # forward
    x0 = Variable(np.array(2))
    # add three times!
    ys = add(add(x0, x0), x0)
    print(ys.data)

    ys.grad = np.array(1)
    ys.backward()
    print(x0.grad)
    
    # 微分をリセットして再計算
    print("clear grad and use x0 again..")
    x0.cleargrad()
    ys = add(x0, x0)
    print(ys.data)
    ys.grad = np.array(1)
    ys.backward()
    print(x0.grad)
    
    # 
    print("test add and square..")
    x = Variable(np.array(2.0))
    y = Variable(np.array(3.0))
    z = add(square(x), square(y))
    z.backward()
    print(z.data)
    print(x.grad)
    print(y.grad)