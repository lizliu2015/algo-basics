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
follow up: 
1. leetcode 1094 https://leetcode.com/problems/car-pooling/
2. LC253 https://leetcode.com/problems/valid-palindrome-ii/


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
1. leetcode 1109: https://leetcode-cn.com/problems/corporate-flight-bookings/
2. leetcode 2132: https://leetcode-cn.com/problems/stamping-the-grid/

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
1. LC344:https://leetcode.com/problems/reverse-string/
2. LC680:https://leetcode.com/problems/valid-palindrome-ii/

# Data Structure
## 1. Linked List
<img width="1035" alt="image" src="https://user-images.githubusercontent.com/13955626/161233738-64e93994-bfaf-4536-b8f2-de91f8c23030.png">

<img width="683" alt="image" src="https://user-images.githubusercontent.com/13955626/161224094-e8d37a8f-b49c-4ecf-9034-4858efe16f5f.png">
declaire few elements:
head: index of linked list's head
e[i]: value of node i
ne[i]: the pointer of node i
idx: store which node has been used

### 1.1 Insert步骤：
1. 把新的node指向下一个node
3. 把上一个node指向新的node
<img width="758" alt="image" src="https://user-images.githubusercontent.com/13955626/161224293-b823d339-690b-4105-a68b-fc7987c2391b.png">

### 1.2 Delete步骤：
1. 把上一个node指向下一个node
3. 把当前node指向null
<img width="765" alt="image" src="https://user-images.githubusercontent.com/13955626/161224506-6cd24e34-447f-47e1-9291-3d959ddca861.png">

### 1.3 Reverse步骤：
1. 从最后一个node开始，把每一个node都指向前一个node
2. 把第一个node指向null
3. 把head指向原来的最后一个node
<img width="768" alt="image" src="https://user-images.githubusercontent.com/13955626/161228794-c17fc089-ac30-4fc1-a784-1f322cbd5460.png">

练习题：
1. LC21: https://leetcode.com/problems/merge-two-sorted-lists/
2. LC46: https://leetcode.com/problems/lru-cache/

### Linked List和Two Pointer结合：

### 1.4 Linked List 找中间节点：
两个指针同向而行，一个每次前进两个节点，一个每次前进一个节点，当第二个指针出界的时候，第一个指针就停留在中间节点。

### 1.5 Linked List找倒数第k个节点：
两个指针先隔开k个位置，每次相同速度向前进，直到第一个指针出界，第二个指针就停在倒数第k个节点

### 1.6 Linked List Recursion
一般用bottom up的方式， 3 steps
1. subproblem result
2. do sth in current level
3. return result

练习题： 
1. LC206 https://leetcode.com/problems/reverse-linked-list/ 

## 2. Stack and Queue
<img width="1005" alt="image" src="https://user-images.githubusercontent.com/13955626/161240261-e4d0da9f-f14a-4ab9-a50f-da153583f361.png">

### Stack支持的操作：
- push 放进元素
- pop 拿出顶端元素
- peek - 查看最顶端元素
- isEmpty - 查看stack是不是空的

### Queue支持的操作
- enqueue 加入元素
- dequeue 拿出元素
- peek 查看最前端元素
- isFull
- isEmpty

习题：
1. LC20 https://leetcode.com/problems/valid-parentheses/

### 单调栈
1.1 什么是单调栈

1.2 如何维护一个单调栈

单调递增栈：在保持栈内元素单调递增的前提下（如果栈顶元素大于要入栈的元素，将将其弹出），将新元素入栈。

单调递减栈：在保持栈内元素单调递减的前提下（如果栈顶元素小于要入栈的元素，则将其弹出），将新元素入栈。
1.3 单调栈有什么规律

单调栈的时间复杂度是O(n)

<img width="634" alt="image" src="https://user-images.githubusercontent.com/13955626/161318240-cd941855-16bf-4ced-8954-2e7b91b6c959.png">

对于将要入栈的元素来说，在对栈进行更新后（即弹出了所有比自己大的元素），此时栈顶元素就是数组中左侧第一个比自己小的元素；

<img width="633" alt="image" src="https://user-images.githubusercontent.com/13955626/161318281-8ef0092d-2701-447e-8df4-98e568a561af.png">

对于将要入栈的元素来说，在对栈进行更新后（即弹出了所有比自己小的元素），此时栈顶元素就是数组中左侧第一个比自己大的元素；
1.4 什么时候使用单调栈

给定一个序列，求序列中的每一个数左边或右边第一个比他大或比他小的数在什么地方；
2. 代码实现

``` python

#读取数组
n = int(input())
nums = list(map(int, input().split()))

#单调栈
deq = []

#开始处理数据
for i in range(len(nums)):

    # 从单调栈中弹出不满足升序的数
    while deq and deq[-1] >= nums[i]:
        deq.pop()

    # 此时要么栈空(没有最小),否则栈顶元素就是最近的最小元素
    if len(deq) != 0:
        print(deq[-1], end = " ") 
    else:
        print(-1, end = " ")

    # 将当前数据加入单调栈中(当前数据一定能够保证单调栈升序)
    deq.append(nums[i])
```
3. 相关练习题

单调栈：leetcode_42_接雨水 (注: 在很多资料中都会把这个题列为单调栈的题目, 但是我觉得用其他的方法会更清晰一些)

单调栈：leetcode_84_柱状图中最大的矩形

单调栈：leetcode_496_下一个更大元素I

单调栈：leetcode_503_下一个更大元素II

单调栈：leetcode_739_每日温度

### 滑动窗口
什么是滑动窗口
在示例中，我们从数组中第一个元素开始遍历，由于窗口的大小是3，因此当遍历到第三个元素时，窗口就形成了。之后，继续遍历元素时，为了保持窗口的大小为3，左侧元素就需要从窗口中剔除。这样使得窗口一直在向右移动，直到考察到最后一个元素结束，这就是所谓的滑动窗口。
<img width="490" alt="image" src="https://user-images.githubusercontent.com/13955626/161319326-63cda7a4-d321-4e6b-bfbd-76abe7c95107.png">

## Trie
用来高效地存储（Insert）和查找（Search）字符串的一种集合的数据结构
<img width="638" alt="image" src="https://user-images.githubusercontent.com/13955626/161321517-9c6a3e33-5acb-418d-a0c2-270520439d24.png">

## Backtrack
一种DFS的形式


## 并查集 Union Find





## 堆 Heap

## Hashmap



# 搜索与图论
## 深度优先搜索 DFS
- 数据结构：stack
- 空间： O(h)
- 最短性：不具有

    深度优先遍历，从初始访问结点出发，初始访问结点可能有多个邻接结点，深度优先遍历的策略就是首先访问第一个邻接结点，然后再以这个被访问的邻接结点作为初始结点，访问它的第一个邻接结点， 可以这样理解：每次都在访问完当前结点后首先访问当前结点的第一个邻接结点。
    我们可以看到，这样的访问策略是优先往纵向挖掘深入，而不是对一个结点的所有邻接结点进行横向访问。
    显然，深度优先搜索是一个递归的过程

使用dfs需要考虑三点： 
1.是否有条件不成立的信息（撞南墙）， 是否有条件成立的信息（到终点）--  if 语句

2，是否需要记录节点（记轨迹）是为了防止重复访问，出现环回
如何标记一个节点是否访问过呢？标记常用方法有数组法和set
```
    bool visited[length] ; //数组表示，每访问过一个节点，数组将对应元素置为true
    Set<类型> set; //建立set，每访问一个节点，将该节点加入到set中去
    栈：用栈来保存当前节点信息，当遍历新节点返回时能够继续遍历当前节点。可以使用递归栈。
```
3. 回朔

回退到上一节点，继续向下搜索

### 啥时候使用dfs呢
当我们遇到的问题与路径相关，且不是寻找最短路径（最短路径为BFS，下次再说），或者需要遍历一个集合中的所有元素，或者是查找某一种问题的全部情况时，我们可以考虑使用DFS来求解。

dfs是经常使用的暴力解法。常常，其优化版本要么是记忆化搜索，要么就是dp。但是因为dfs很好想，而且很好实现（只要递归即可），所以dfs常常出现在题解中。
### 如何用dfs？

    dfs出现的地方，常常对应的是多个选择的地方，选或者不选；放或者不放；做或者不做…等等。再上一个选择的基础上，又迎来下面的一波选择。
    为了避免重复选择，通常我们会用vis[]数组标记当前这轮深搜时的选择情况，但是需要注意的是，在当前这轮结束的时候，通常（几乎就是所有的情况）就需要将
    这个vis数组清除做的标记，因为，别的情况可能会因为这个标记而做出错误选择。即，应该保证：
    某轮做的标记，在该轮结束时，应该予以清除。需要做出选择的时候，通常就是调用dfs的时候。

### 什么时候dfs在for循环里，什么时候又不在？判断的依据是什么？

for循环也是一种选择的体现，比如说一个顶点有很多条分支，这时就可以使用for依次遍历各个分支，然后把这个分支所在的节点作为参数传递到dfs中，进行下一次的深搜。

    在for循环中的dfs通常用于图的遍历。
    直接dfs的情况通常在于非图的遍历。
-----------------------------------
©著作权归作者所有：来自51CTO博客作者lawsonabs的原创作品，请联系作者获取转载授权，否则将追究法律责任
dfs刷题模板总结
https://blog.51cto.com/lawsonabs/3055202

### 优化方法
记忆化搜索就是：将一次搜索的结果放在一个容器中存储下来。等到下次再搜索到这个点时，就应该直接返回之前搜索的值，而不是再往下搜索了。
剪枝
常见的剪枝操作，主要分为两类：
一类是针对题意做的剪枝，就是题目潜藏的存在一种更优解的顺序（这可以参考络谷 P3956 棋盘 这道题。我们怎么走棋盘是有顺序可言的。走了一条，就不用走另一条了。这样就可以成倍的减少dfs）。
第二类就是针对结果的剪枝，如果已经得出了最优解，但是当前深搜的解已经不比这个解更好，那么就应该主动放弃搜索。

### 双向DFS
对于双向DFS，我们考虑看看最短路，起点做一下搜索，记录一下到所有点的距离，终点做一下搜索，记录一下到所有点的距离，那么起点到任一点的距离加上终点到任一点的距离那不就是起点到终点经过这一点的最短距离，我觉得BFS也可以实现，所以在我眼里BFS相对于DFS更强一点，只有说得到特定的某一结果的时候深搜可能会好一点。

### Topology Sort
这一节(以及上一节)参考这个非常棒的视频: https://class.coursera.org/algo-003/lecture/52

拓扑排序是一个dfs的应用, 所谓拓扑排序是指在一个DAG(有向无回路图)里给每个节点定义一个顺序(v1…vn), 使得按照这个顺序遍历的节点, 每一个节点vi都是之前遍历过的的节点(v1 ~ vi-1)所指向的(或没有任何其他节点指向的).

好像还没说清楚… 拓扑排序的一个应用就是对于各种依赖性(比如学习课程A需要先学习过课程B)组成的图寻找一个节点遍历的顺序使其可行.

propositions:

        拓扑排序的结果不唯一.
        有回路的图不存在拓扑顺序.
        如果一个节点没有出边, 那么它可以放在拓扑排序的最后面(没有节点以来它).
        如果一个节点没有入边, 那么它可以放在拓扑排序的最后面.

简单修改一下递归的dfs就可以处理拓扑排序: 维护一个计数器K(初始化为n=所有节点数), 每当一个点已经遍历完毕(所有通过这个点可以到达的点都已经被走过)以后, 就把这个点的顺序设为K, 同时减少K.  为了给所有点拓扑排序, 只要从一个没有出边的节点出发进行遍历, 一直运行到所有的节点都已经访问过为止。

### Backtrack 回溯




### 模版
一维：
``` python
#数组全排列
def dfs(u):

    if u == n:    #当所有坑位被占满 那么输出储存的路径
        for i in range(0,n):
            print(p[i],end=" ")
        print('')

    for i in range(1,n+1):
        if not st[i]: #确认数字状态，是否已经被使用 如果没有被占执行下面操作
            p[u] = i  #在坑位上填上次数字
            st[i] = True #标注数字状态，已经被使用
            dfs(u+1) #进入下一层
            st[i] = False #回溯恢复数字状态

if __name__ == '__main__':
    N = 10
    p, st= [0] * N,[False] * N

    n = int(input())

    dfs(0)
``` 
二维：
```python
n = int(input())
dg, ndg, col_unused = [True] * (n<<1) , [True] * (n<<1), [True] * n
row = [-1] * n

def dfs(k):
    if k == n:
        [print('.'*i + 'Q' + '.' * (n - i - 1)) for i in row] + [print('')]
        return 

    for i in range(n):
        if col_unused[i] and dg[i + k] and ndg[n + i - k]:
            row[k] = i
            col_unused[i] = dg[i + k] = ndg[n + i - k] = False
            dfs(k + 1)
            col_unused[i] = dg[i + k] = ndg[n + i - k] = True

dfs(0)
``` 

练习题：
1. LC51: https://leetcode.com/problems/n-queens/solution/
2. LC200
3. LC130
4. LC133
5. LC329
6. LC417


## 宽度优先搜索 BFS
- 数据结构：queue
- 空间： O(2^h)
- 最短性：具有“最短路”



