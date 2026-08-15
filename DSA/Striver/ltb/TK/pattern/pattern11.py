def printpattern(n):
    temp=1
    for i in range(n+1):
        for j in range(i):
            # print("*",end="")
            if(temp==1):
                print(temp,end="")
                temp=0
            else:
                temp=0
                print(temp,end="")
                temp=1
        print("")            
printpattern(5)

#suceed but the best apporach is 
# def printpattern(n):
#     temp = 1

#     for i in range(n + 1):
#         for j in range(i):
#             print(temp, end="")
#             temp = 1 - temp
#         print()

# printpattern(3)

# # print(11%2)

#just trying the alterating sequence
# we can print like j%2 because the numbers are like even odd even so the sequecen would be like 0,1,0,0
# this would fail it whould not work because they asked like 0,1,0,1 like it should even contuine even with the next row


# def printpattern(n):
#     for i in range(n+1):
#         for j in range(1,i+1):
#             # using the i lenght odd or even thing
#             if(j%2==0):
#                 print("0",end="")
#             else:
#                 print("1",end="")    
#         print("")            
# printpattern(5)