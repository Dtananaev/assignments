#lists
xs=[3,1,2]
print xs, xs[2]
print xs[-1]
xs[2]="foo"
print xs
xs.append("bar")
print xs
x=xs.pop()
print x
print xs

#slicing

nums=range(5)
print nums
print nums[2:4]
print nums[2:]
print nums[:2]
print nums[:-1]
nums[2:4]=[8,9]
print nums

# loops
animals= ['cat','dog', 'monkey']
for animal in animals:
    print animal


for idx, animal in enumerate(animals):
    print '#%d: %s' %(idx+1,animal)

#list comprehensions

nums = [0,1,2,3,4]
squares=[]
# naive
for x in nums:
    squares.append(x**2)
print squares
#list comprehension
nums = [0,1,2,3,4]
squares=[x**2 for x in nums]
print squares
#list comprehension with condition
nums = [0,1,2,3,4]
squares=[x**2 for x in nums if x%2==0]
print squares



nums = [0,1,2,3,4]
a=[1,1,1,2,2,2,2]

import numpy as np
a=np.array([1,2,3,4,5,6,7,8,9,99])
print a
print a[nums]




