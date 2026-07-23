# A while loop repeats a block of code as long as a condition is True.

i=1
while i<=5:
    print(i)
    i+=1

#if the loop condtion is flase intllilay
i = 10

while i <= 5:
    print(i)

before exeuting the body the while loop check the condition
i =1
while i<=5:
    print(i) #infite loop

important 
initlize- check -execute-update

#common use case
number = int(input("enter postive number"))

while number<=0:
    number =int(input("try entering valid number"))
    if not number:
            break # when we want to sometimes stop the loop
    else:
         continue #skip the rest of the current iteration
    

print("valid number")


for vs while
when we know how many times to repeat for is best
when we don't know how many times to repeat while is best

# You'll use while loops in many important algorithms, such as:

Binary Search
Two Pointers
Fast & Slow Pointers
Linked List Traversal
Tree Traversal (iterative)
Reading input until a condition is met
Game loops and menu-driven programs