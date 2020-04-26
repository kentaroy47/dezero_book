# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:08:05 2020

@author: arutema47
"""

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
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            
            if x.creator is not None:
                funcs.append(x.creator)
        
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

if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()
    
    # forward
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    
    # Backwards
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)