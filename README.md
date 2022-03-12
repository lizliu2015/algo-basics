# algo-basics

2. Merge Sort
(归并排序) O(nlog(n)

自底向上的方法，先划分为两个子区间，然后分别对两个子区间排序，再将排序好的子区间进行合并。

```python
def quicksort(nums, start, end):
    if start >= end :
        return 
    left, right = start, end
    pivot = nums[(start + end) // 2]

    while left <= right:
        while left <= right and nums[left] < pivot:
            left += 1
        while left <= right and nums[right] > pivot:
            right -= 1
        if left <= right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -=1
    quicksort(nums, start, right)
    quicksort(nums, left, end)
```
follow up :
kth largest number (quick select)
find median sort arrays( based on kth, 2 pointers, divide and conquer )
