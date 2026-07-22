# #let's learn

# #inputs & outputs

# print("hello")
# print("we can print result expression")

# name  = input("Enter your name\n")


# #printing variables
# x =40
# print(x)


# #printing muliple values
# x =150
# print(x,40,name,"krishna",)


# #to print on the same line we this end = ""
# print("hello",end=" ")
# print("world")

# #taking input
# message = input()
# print(message)

# x = input()
# print(x)


# print(type(x))


# integer =  int(input())
# print(type(integer))

# price = float(input())
# print(price)


# #dobule inputs 
# x,y = input().split()
# print(x)
# print(y)


#the diff between this and the above is this coversta the list into list
x , y =  map(int,input().split())
print(x)
print(y)


arr = list(map(int,input().split()))
print(arr)

name = input("Enter your name")

print(name)

#serpator
print(1,2,3,sep="-")

#end parameter
for i in range(5):
    print(i,end="")

# 4. Taking Multiple Lines of Input
n = int(input())
for _ in range(n):
    x =int(input())
    print(x)


# you only want the first character:

ch = input()[0]

print(ch)

