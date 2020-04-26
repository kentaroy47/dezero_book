# -*- coding: utf-8 -*-
import numpy as np

class Variable():
    def __init__(self, x):
        self.data = x
        self.grad = None
        self.creator = None
    
    # for define-by-run
    def set_creator(self, func):
        self.creator = func
        
    # for auto-grad
    def backward(self):
        f = self.creator # 関数を取得
        if f is not None:
            x = f.input # 入力を取得
            x.grad = f.backward(self.grad) # 関数のbackwardメソッドを呼ぶ
            x.backward() # 一つ前の変数のbackwardを呼ぶ
        
class Function ():
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        # set output for backpropagation
        output = Variable(y)
        # set the creator of the output, which is the function itself.
        output.set_creator(self)
        # remember input, outputs
        self.input = input
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
        
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def numerical_diff(f, x, eps=1e-5):
    """
    f..微分を求める元の関数
    x..微分を求める値の中心
    eps..微分計算時のeps
    
    return..微分結果
    """
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    diff = (y1.data - y0.data) / (2 * eps)
    return diff

def f2(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))
