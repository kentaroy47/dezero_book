# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:08:05 2020

@author: arutema47
"""

import numpy as np
import unittest
from dezero import numerical_diff

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
        
    # for auto-grad
    def backward(self):
        # for initial grad
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        # define-by-run in a loop fasion
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
        output = Variable(as_array(y))
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

## utility functions ##
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

## Test functions ##
class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
        
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
        
    def test_gradient_check(self):
        x = Variable(np.random.rand(100)) # generate 100 random inputs
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        # check if the two computed grads match within error margin
        flg = np.allclose(x.grad, num_grad) 
        self.assertTrue(flg)

if __name__ == "__main__":    
    # forward
    x = Variable(np.array(0.5))
    a = square(x)
    b = exp(a)
    y = square(b)
    
    # Backwards
    y.backward()
    print(x.grad)

# commentout if renning tests    
unittest.main()