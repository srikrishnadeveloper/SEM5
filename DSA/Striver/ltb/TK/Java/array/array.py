#in python we call the array as list
#liner data structure
    # 89 → 76 → 91 → 65 → 80
    #everything is arraged one after another
#index = every element has postion
# python is  "0" zero based index because we does't needed to subtract one to find the elment when calculating the memory address
#python array can grow larger automicallty not like the c++ array


name = "Krishna"

# "K r i s h n a"
# 0 1 2 3 4 5 6

print(name[5])
print(len(name))

#strings are immutable ie is once its value setted it can't be changed only by charcater by charcter change is not allowed ie name[2] = "k"


arr =[1,2,3,4,5]
arr.append(6)
arr.insert(2,9)
arr.remove(9)
arr.pop(2)
print(30 in arr) # traversak
print(arr.index(30))


