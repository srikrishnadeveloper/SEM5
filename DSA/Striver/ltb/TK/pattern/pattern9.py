def printpattern(n):
    for i in range(n):
        print(" "*(n-i-1)+"*"*(2*i+1)+" "*(n-i-1))
    for i in range(n,0,-1):
        print(" "*(n-i+1)+"*"*(2*i-1)+" "*(n-i+1))
printpattern(3)