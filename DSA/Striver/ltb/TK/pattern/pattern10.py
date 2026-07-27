def printpattern(n):
    for i in range(n+1):
        print("*"*i)
        if(i==n):
            for i in range(n-1,0,-1):
                print("*"*i)
printpattern(3)