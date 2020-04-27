from enum import Enum
class units(Enum):
    INCHES=1
    MM=2

class PaperMedia:
    def __init__(self,W,H,unit=units.INCHES):
        self.DPI=300
        self.unit = unit
        self.W=W
        self.H=H

    def canvas(self):
        assert self.unit == units.INCHES
        return (self.W*self.DPI,self.H*self.DPI)

    def size(self):
        return (self.W,self.H)

    @property
    def aspect(self):
        return self.W/self.H

    def scale(self,w,h):
        return (w,h*self.aspect())

class A4(PaperMedia):
    def __init__(self):
        super(A4,self).__init__(W=8.27,H=11.69,unit=units.INCHES)

class USLetter(PaperMedia):
    def __init__(self):
        super(USLetter,self).__init__(W=8.5,H=11,unit=units.INCHES)
