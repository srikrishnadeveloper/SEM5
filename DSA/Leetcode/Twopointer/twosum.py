class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        
        left = 0
        right  = len(numbers)-1

        while(left<right){
            int sum = numbers[left]+numbers[right]
            if(sum==target){
                return new int []{left+1,right+1}

            }
            else if (sum>target){
                right--;
            }

        }