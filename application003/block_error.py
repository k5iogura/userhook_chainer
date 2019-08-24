def func(mat, x, y, num, num2):  #かたまりでエラーを注入する
    import numpy as np # NumPyモジュールをインポート
#    d = mat.data
    d = mat.copy()                      # make sure copying
    n,h,w,c = mat.shape                 # n:batch-size
    print('in blk_err num=',num)
    for b in range(n):                  # enhance for batch calc.
        for i in range(num):
            for j in range(num):
                d[b,i+x,j+y,0]=num2     # enhance for batch calc.
#                d[0,i+x,j+y,0]=num2
#                print('err_loc:[%d, %d]' %(i+x,j+y))
    return(d)
