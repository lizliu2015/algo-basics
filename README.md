# Table of Contents
1. [Sorting](#Sorting)
2. [Binary Search](#BinarySearch)
3. [Presum](#Presum)
4. [Two Pointers](#TwoPointers)

## 1. Sorting <a name="Sorting"></a>
### 1.1 Quick Sort
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

### 1.2 Merge Sort
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

## 2. Binary Search <a name="BinarySearch"></a>
综合使用课上所讲的两种二分模板；
1. 找左边的最后一个：
定义性质: 左边元素满足<=target , 结果是查找范围的右边界;
2. 找右边的第一个：
定义性质: 左边元素满足< target （即右边元素>=target ），结果是查找范围的左边界;

注意:
1. python除法不会自动向下取整, 需要手动int();
2. 二分法一定会输出结果, 但这个结果不一定是满足要求的, 需要额外判断;

```python
# 数组共有n个数, 查询q次
n, q = map(int, input().split())

# 读取数组
nums = list(map(int, input().split()))


def get_right(target): # 获取左边数组的右边界

    l = 0
    r = n - 1

    while l < r:
        mid = int((l + r + 1) / 2)
        if (nums[mid] <= target): # 左边数组的性质: <= target
            l = mid
        else:
            r = mid - 1

    return l

def get_left(target):  # 获取右边数组的左边界

    l = 0
    r = n - 1

    while l < r:
        mid = int((l + r) / 2)
        if (nums[mid] >= target): # 右边数组的性质：>= target
            r = mid
        else:
            l = mid + 1

    return l 


# q次查询
for i in range(q):

    target = int(input())

    left = get_left(target)
    right = get_right(target)

    if nums[left] == target and nums[right] == target:
        print(get_left(target), get_right(target))
    else:
        print(-1, -1)
```
备注: 
二分练习题：
1. leetcode_704_二分查找 https://leetcode.com/problems/binary-search/
2. https://leetcode.com/problems/random-pick-with-weight/
3. https://leetcode.com/problems/find-peak-element/



## 3. Presum/差分 <a name="Presum"></a>

差分实质上就是前缀和的逆运算，主要解决连续多次在部分区间增加某个值 c 之后，求更新之后的数组。

问题：

输入一个长度为n的整数序列。接下来输入m个操作，每个操作包含三个整数l, r, c，表示将序列中[l, r]之间的每个数加上c。请你输出进行完所有操作后的序列。

输入样例：

```
6 3 
1 2 2 1 2 1 
1 3 1 
3 5 1 
1 6 1 
```

输出样例：

```3 4 5 3 4 2```

数据范围
```
1≤n,m≤100000,
1≤l≤r≤n,
−1000≤c≤1000,
−1000≤整数序列中元素的值≤1000
```

问题求解

该题最直观的思想是针对每次操作遍历一次数组，然后在相应的位置增加 c ，该方法的时间复杂度为 O(mn)

。现在我们采用差分的方法来进行算法的优化。

对于一个数组 b[n+1] ：

```([x, x, x, l, x, x, r, r+1, x, n])```

如果在 b[l] 位置加上 c ，则从 b[l] 开始的所有前缀和数组都会加上 c ；同理，为了保证只在 [l,r] 的区间增加 c ，我们需要将 b[r+1]-=c 。最终我们只需要根据得到的数组 b 求前缀和即可。采样该方法对于每次的区间操作只需要 O(1) 的时间复杂度，而构造差分矩阵的时间复杂度为 O(n) ，因此总共的时间复杂度为 O(n) 。

核心代码：
```python 
def insert(b, l, r, c):
    b[l] += c # add c to all int after l
    b[r+1] -= c #minus c to all int after r+1

if __name__ == "__main__":
    n, m = map(int, input().split())
    a = [0] * (n + 10)  # 原值数组
    b = [0] * (n + 10)  # 差分数组
    nums = list(map(int, input().split()))
    for index, val in enumerate(nums):
        a[index+1] = val 
    for i in range(1, n+1):  # 强烈建议都从 1 开始
        insert(b, i, i, a[i])
    while m > 0:
        m -= 1
        l, r, c = map(int, input().split())
        insert(b, l, r, c)
    for i in range(1, n+1):
        b[i] += b[i-1]
    for i in range(1, n+1):
        print(b[i], end=" ")
```
follow up: leetcode 1094 https://leetcode.com/problems/car-pooling/


### 差分矩阵

差分矩阵实质上就是二维上的差分：

问题：

输入一个n行m列的整数矩阵，再输入q个操作，每个操作包含五个整数x1, y1, x2, y2, c，其中(x1, y1)和(x2, y2)表示一个子矩阵的左上角坐标和右下角坐标。每个操作都要将选中的子矩阵中的每个元素的值加上c。请你将进行完所有操作后的矩阵输出。

输入样例：
```
3 4 3  # n, m, q
1 2 2 1
3 2 2 1
1 1 1 1
1 1 2 2 1  # q 个操作
1 3 2 3 2
3 1 3 4 1
```
输出样例：
```
2 3 4 1
4 3 4 1
2 2 2 2
```
问题求解

直接借鉴一维差分和二维矩阵前缀和的思想，对于要在某个区间增加一个值 c ，我们需要对差分矩阵 b[n+1][m+1] 进行如下操作：
```
b[x1][y1] += c
b[x2+1][y1] -= c
b[x1][y2+1] -= c
b[x2+1][y2+1] += c
```
二维差分模板，主要应用于在 某一个区间内加上一个固定的值，该方法能将O(n)的时间复杂度降为O(1)

核心代码：

```python
def insert(b, x1, y1, x2, y2, c):
    b[x1][y1] += c
    b[x2+1][y1] -= c
    b[x1][y2+1] -= c
    b[x2+1][y2+1] += c

if __name__ == "__main__":
    n, m, q = map(int, input().split())
    N = 1010  # 创建更大的矩阵，省去边界条件的考虑
    a = [[0] * (N) for i in range(N)]
    b = [[0] * (N) for i in range(N)]
    for i in range(1, n+1):
        nums = map(int, input().split())
        for j, val  in enumerate(nums):
            a[i][j+1] = val
    for i in range(1, n+1):
        for j in range(1, m+1):
            insert(b, i, j, i, j, a[i][j])
    # q 次操作
    while q > 0:
        q -= 1
        x1, y1, x2, y2, c = map(int, input().split())
        insert(b, x1, y1, x2, y2, c)
    # 求前缀和
    for i in range(1, n+1):
        for j in range(1, m+1):
            b[i][j] = b[i-1][j] + b[i][j-1] - b[i-1][j-1] + b[i][j]
    # 输出
    for i in range(1, n+1):
        for j in range(1, m+1):
            print(b[i][j], end=" ")
        print()
```

follow up 
leetcode 1109: https://leetcode-cn.com/problems/corporate-flight-bookings/
leetcode 2132: https://leetcode-cn.com/problems/stamping-the-grid/

## 4.Two Pointer <a name="TwoPointers"></a>

```python
n=int(input()) # 输入n和整个序列
a=list(map(int,input().split()))

s=[0]*(n+1) # s[] 存储当前序列中每个字符出现的次数

res=0 # 结果

for i in range(n):
    s[a[i]]+=1          # 当前字符出现的次数+1
    while s[a[i]]>1:    # 如果当前字符出现次数>1,表示有重复，需要进行处理
        s[a[j]]-=1      # 让j开始往右走，走一次，看一次是否还有重复
        j+=1
    res=max(res,i-j+1)  # 更新答案

print(res)
```
practice:
LC344:https://leetcode.com/problems/reverse-string/


