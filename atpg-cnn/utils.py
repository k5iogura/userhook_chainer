from datetime import datetime as dt
# time stamp utility
class timestamp:
    def __init__(self, msg):
        self.start = dt.now()
        print('*', msg, '{} #'.format(self.start))
    def ts(self,obj):
        return '{}/{}/{}:{}:{}:{}'.format(obj.year,obj.month,obj.day,obj.hour,obj.minute,obj.second)
    def click(self, msg=''):
        now = dt.now()
        print('*',msg,'{} - {} = {} #'.format(self.ts(now), self.ts(self.start), now - self.start))
        return now

# << Pattern Name manager for tableX >>
class PatNames:
    def __init__(self, batch):
        self.offset= 0
        self.batch = batch
        self.unqID = [None] * self.batch
        self.count = 0
        self.outix = 0
    def index2name(self, pattern_index):
        if self.unqID[pattern_index] is None:
            self.unqID[pattern_index]  = self.count
            self.count += 1
            return self.count - 1
        else:
            return self.unqID[pattern_index]
    def add(self,pattern_index):return self.index2name(pattern_index)
    def name2index(self, unqID):return self.unqID.index(unqID)
    def extend(self,batch):
        self.unqID.extend([None] * batch)
        self.offset = self.batch
        self.batch += batch
        print('* PatName.extend offset={} extend={} size={} valid={}'.format(
            self.offset, batch, self.batch, self.count))
    @property
    def table(self):return self.unqID
    def __iter__(self):
        self.outix = 0
        return self
    def __next__(self):
        if self.outix >= self.batch: raise StopIteration
        for idx in range(self.outix, self.batch):
            unqID = self.unqID[ idx ]
            if unqID is not None: break
        self.outix = idx + 1
        if unqID is None: raise StopIteration
        return idx, unqID

