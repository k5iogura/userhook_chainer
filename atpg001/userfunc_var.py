class VAR:
    __os= 1  # 1:linux 0:Windows
    __n = 0

    @property
    def n(self):return VAR.__n
    @n.setter
    def n(self,val):VAR.__n=val

    @property
    def os(self):return VAR.__os
    @os.setter
    def os(self,val):VAR.__os=val

