class VAR:
    __os= 1  # 1:linux 0:Windows
    __n = 0
    __device = -1 # -1:cpu 0:gpu

    @property
    def n(self):return VAR.__n
    @n.setter
    def n(self,val):VAR.__n=val

    @property
    def os(self):return VAR.__os
    @os.setter
    def os(self,val):VAR.__os=val

    @property
    def device(self):return VAR.__device
    @device.setter
    def device(self,val):VAR.__device=val

