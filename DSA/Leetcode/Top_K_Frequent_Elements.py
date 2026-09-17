nums = [1,1,1,2,2,3]
nums.sort()#sorted the values
dic = {}
for num in nums:
    if num in dic:
        pass
    else:
        dic[num] = nums.count(num)
sorted_dic = {}
sorted_dic = sorted(dic.items(),key=lambda x:x[1],reverse=True)
result = []
for i in range(2):
    result.append(sorted_dic[i][0])
print(result)