class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        # [1, 2, 3, 4, 5]
        # [100, 3, 7, 88]

        # loop through array with two pointers
        # check sum, if it equals target, return pointer indices
        # else continue
        # concern -- double counting same index

        for i, i_val in enumerate(nums[:-1]):
            for j, j_val in enumerate(nums[i+1:]):
                if i_val + j_val == target:
                    return [i, j + i + 1]
        return None


# print(Solution().twoSum([3,2,4], 6))


class SearchInsertPosition(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """

        # brute force: loop through array w/ enumerate
        # if target found, return index, else return previous index

        # divide and conquer alternative
        # recursive
        # is median above, below or equal to target?
        # split appropriate portion and repeat


        if len(nums) == 0:
            return 0

        def get_mid(sub):
            mid = int(len(sub) / 2)
            number = sub[mid]
            if number == target:
                return nums.index(number)
            elif target > number:
                if len(sub) == 1:
                    return nums.index(number) + 1
                return get_mid(sub[mid:])
            elif target < number:
                if len(sub) == 1:
                    return nums.index(number)
                return get_mid(sub[:mid])

        return get_mid(nums)

print(SearchInsertPosition().searchInsert([4, 5, 6], 5))