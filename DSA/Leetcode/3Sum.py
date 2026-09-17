nums = [-100,-70,-60,110,120,130,160]
if len(nums)<3:
    print([])
result = []
l=0
r=len(nums)-1
temp=[]
while(l<r):
    while (r>l):
        for i in range(l+1,r):
            if i!=(l or r) and l!=r and r!=l:
                if (nums[l]+nums[r]+nums[i])==0:
                    temp = [nums[i],nums[r],nums[i]]
                    temp.sort()
                    if temp not in result:
                        result.append(temp)
        r-=1
    l+=1
print(result)


if len(nums)<3:
            return []
        result = []
        l=0
        r=len(nums)-1
        temp = []
        while(l<r):
            for i in range(l+1,r):
                    if (nums[l]+nums[r]+nums[i])==0:
                        temp = [nums[i],nums[r],nums[l]]
                        temp.sort()
                        if temp not in result:
                            result.append(temp)
            l+=1
            r-=1
        return result      




        class Solution:
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        result=[]
        temp = []
        for i in range(len(nums)):
            l=i+1
            r =len(nums)-1
            while(l<r):
                if (nums[i]+nums[r]+nums[l])==0:
                    temp = [nums[i],nums[r],nums[l]]
                    temp.sort()
                    if temp not in result:
                        result.append(temp)
                l+=1
        return result   