nums = [-1,0,1,2,3]
prefix = 1
suffix = 1
boo = True
answer = [1] * len(nums)


prefix =1
for i in range(len(nums)):
    answer[i] = prefix
    prefix = prefix*nums[i]
suffix = 1
for j in range(len(nums)):
    answer[i]= suffix
    suffix = suffix*nums[i]
print(answer)   
# print(result)








# boo = False
# for i in range(len(nums)):
#     temp = 0
#     for j in range(len(nums)):
#         if i!=j:
#             if temp==0 and boo==False:
#                 temp = nums[j]
#                 boo = True
#             else:
#                 temp = (temp*nums[j])
#     answer.append(temp)
#     boo = False
# print(answer)









# for i in range(len(nums)):
#     # print(i)
#     temp = 1
#     for j in range(len(nums)):
#         if i!=j:
#             temp = (temp*nums[j])
#         else:pass
#     answer.append(temp)
# print(answer)