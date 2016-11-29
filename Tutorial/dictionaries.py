#Dictionaries

d={'cat': 'cute','dog':'furry'}
print d['cat']
print 'cat' in d
d['fish']='wet'
print d['fish']
print d.get('monkey', 'N/A')
print d.get('fish', 'N/A')
del d['fish']
print d.get('fish','N/A')


#loop
d={'person':2, 'cat':4, 'spider':8}
for animal in d:
    legs=d[animal]
    print 'A %s has %d legs' % (animal, legs)
#to acess the key
for animal, legs in d.iteritems():
    print "A %s has %d legs" %(animal, legs)    

#Dictionary comprehension
nums=[0,1,2,3,4]
even_nmbr_sqr={x:x**2 for x in nums if x%2==0}
print even_nmbr_sqr
