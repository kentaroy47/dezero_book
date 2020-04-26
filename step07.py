# -*- coding: utf-8 -*-
import dezero
import numpy as np

if __name__ == "__main__":
    A = dezero.Square()
    B = dezero.Exp()
    C = dezero.Square()
    
    # forward
    x = dezero.Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    
    # Backwards
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)