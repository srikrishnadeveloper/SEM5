def printpattern(n):
        for i in range(n):
            print(" "*(n-i-1)+"*"*(2*i+1)+" "*(n-i-1))
printpattern(3)



# (N - i - 1) spaces on the left (to center align the stars),
# (2 * i + 1) stars in the middle,
# (N - i - 1) spaces on the right.

