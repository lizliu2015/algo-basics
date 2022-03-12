# algo-basics
1. Quick Sort
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

2. Merge Sort
(归并排序) O(nlog(n)

自底向上的方法，先划分为两个子区间，然后分别对两个子区间排序，再将排序好的子区间进行合并。

```python
n = int(input())
list1 = list(map(int, input().split()))

def merge_sort(list1):
    if len(list1) <= 1:
        return
    mid = len(list1) // 2
    L = list1[:mid]
    R = list1[mid:]
    merge_sort(L)
    merge_sort(R)

    i = j = k = 0
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            list1[k] = L[i]
            i += 1
        else:
            list1[k] = R[j]
            j += 1
        k += 1
    while i < len(L):
        list1[k] = L[i]
        k += 1
        i += 1
    while j < len(R):
        list1[k] = R[j]
        k += 1
        j += 1
```

