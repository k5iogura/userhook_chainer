def mylist():
    mylist = []
    for l in range(7):
        for k in range(7):
            for i in range(0,28,7):
                for j in range(0,28,7):
                    x=i+l
                    y=j+k
                    mylist.append((x,y))
    return(mylist)

#d= mylist()
#n=85
#print(d[n-1])
#print(d[n-1][0])
#print(d[n-1][1])
