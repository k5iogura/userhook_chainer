def mylist():
    mlist=[]
    for i in range(7):
        for j in range(7):
            for k in range(0,28,7):
                for l in range(0,28,7):
                    mlist.append([i+k,j+l])
#    for i in range(7*7*4*4):
#        print(mlist[i])
    return mlist
if __name__=='__main__':
    mylist()
