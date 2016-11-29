class Greeter(object):
    #constructor
    def __init__(self,name):
        self.name=name

    #Instance method
    def greet(self, loud=False):
        if loud:
            print "HELLO, %s!" % self.name.upper()

        else:
            print "Hello, %s" % self.name

g=Greeter('Den')
g.greet()
g.greet(loud = True)
