nums = [0,3,7,2,5,8,4,6,0,1]
nums.sort()
temp = nums[0]
count=1
# print(temp)
for i in range(1,len(nums)):
    # print(nums[i])
    if ((temp+1)==nums[i]):
        count+=1
        temp=nums[i]
print(count)
