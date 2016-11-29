animals ={'cat', 'dog','cat'}
print animals
print 'cat' in animals
print'fish' in animals
animals.add('fish')
print 'fish' in animals
print len(animals)
animals.add('cat')
print len(animals)
animals.remove('cat')
print len (animals)
#loops
animals ={'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print '#%d: %s' % (idx+1, animal)
from math import sqrt
nums={int (sqrt(x)) for x in range(30)}
print nums
