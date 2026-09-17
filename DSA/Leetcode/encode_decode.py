# program two function one encode the list of string and then the other decode the list of the string       
strs = ["Hello","World"]
# class Solution:
#     def encode(self, strs: List[str]) -> str:
#         big_string=""
#         for i,enum in enumerate(strs):
#             big_string +="|"+enum
#         self.decode(big_string)
#     def decode(self, s: str) -> List[str]:
#         lis = []
#         lis = str.split("|")
#         return lis[1:]
# print(Solution.encode(strs))
class Solution:
    def encode(self,strs):
        big_string = ""
        for word in strs:
            big_string+="|&|"+word
        sol = Solution
        sol.decode(str,big_string)
    def decode(self,st):
        lis= []
        lis=st.split("|&|")
        print(lis[1:])
Solution.decode(strs)