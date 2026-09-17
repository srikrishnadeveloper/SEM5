strs = ["eat","tea","tan","ate","nat","bat"]
dic = {}#intlizeing the dic
#main loop
result = []
for i in range(len(strs)):
    key_sorted = "".join(sorted(strs[i]))
    if key_sorted not in dic:
        dic[key_sorted] = [strs[i]]
    else:
        dic[key_sorted].append(strs[i])
for key in dic:
    result.append(dic[key])
print(result)