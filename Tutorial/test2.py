import numpy as np


a=np.array([[1,2,3,4],[4,5,6,4], [7,8,9,4]])

b=np.array([0,1,2])
d=np.tile(b,[4,1])
print ("d",d)

c=a[b,b]
print a
print b
print b.shape

print("c", c)
