class runtimeviz():
    vars = dict()
    lnks = dict()
    _fin    = False
    def __init__(self):
        runtimeviz.vars = dict()
        runtimeviz.lnks = dict()
        runtimeviz._fin    = False

    def regist_lnk(self, name, link):
        if runtimeviz._fin:return link
        runtimeviz.lnks[name]= link
        return link

    def regist_var(self, name, chain):
        if runtimeviz._fin:return chain
        runtimeviz.vars[name]= chain
        return chain

    def regist_end(self):
        runtimeviz._fin=True

    def list(self):
        if runtimeviz._fin:return
        for k in runtimeviz.vars.keys():
            print(k,runtimeviz.vars[k].data[0].shape)
        for k in runtimeviz.lnks.keys():
            print(k)
