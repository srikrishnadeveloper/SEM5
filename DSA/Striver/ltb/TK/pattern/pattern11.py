# Intuition: This pattern prints alternating 1s and 0s in each row, starting with 1 on even-indexed rows and 0 on odd-indexed rows. The value alternates after each print using basic toggling logic.

# Take an integer N as input representing the number of rows.
# Loop from 0 to N-1 to handle each row.
# If the row index is even, set the starting value to 1; otherwise, set it to 0.
# For each row, print i+1 numbers while toggling the value between 1 and 0 after each print.
# After printing each row, move to the next line.

def printpattern(n):
    for i in range(0,n-1):
        if((i%2)==0):
            for i in range(n):
                print()
        else:

printpattern(5)

# temp =7
# if ((temp%2)==0):
#     print("even")
# else:
#     print("odd")