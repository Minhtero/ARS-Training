# Notebook: Các thuật toán Python cho phỏng vấn AI Engineer
# Phiên bản: từng cell một, mỗi bài có hướng dẫn chi tiết, giả thuyết, mã, test, mẹo
# Lưu ý: mỗi cell bắt đầu bằng '# %%' để chạy trong Jupyter/VSCode.

# %%
"""
HƯỚNG DẪN SỬ DỤNG
- Chạy từng cell (Shift+Enter) trong Jupyter hoặc chạy file .py trong VSCode từng block.
- Mỗi bài được trình bày: Mục tiêu -> Ý tưởng chính -> Bước giải (pseudocode/invariant) -> Mã Python -> Ví dụ chạy -> Edge cases -> Lời giải mở rộng / biến thể -> Mẹo phỏng vấn.
- Nếu muốn mình có thể xuất thành .ipynb, thêm testcases tự động hoặc mock-interview flow.
"""

# %%
# Cell 1: Longest substring without repeating characters (Sliding Window)
# ----- MỤC TIÊU -----
# Trả về độ dài lớn nhất của một substring không có ký tự lặp trong chuỗi s.
# ----- Ý TƯỞNG CHÍNH -----
# Dùng kỹ thuật sliding window với hai con trỏ l (bắt đầu window) và r (kết thúc window).
# Dùng map/dict lưu lần xuất hiện cuối cùng của ký tự. Khi gặp ký tự đã xuất hiện trong window,
# di chuyển l tới vị trí sau lần xuất hiện trước đó để đảm bảo window không có ký tự lặp.
# ----- INVARIANT -----
# Window [l..r] luôn là substring không có ký tự lặp.
# ----- ĐỘ PHỨC TẠP -----
# Time: O(n), Space: O(min(n, charset))

def longest_unique_substr(s: str) -> int:
    """Trả về độ dài lớn nhất của substring không có ký tự lặp."""
    # last[c] = last index where c appeared
    last = {}
    l = 0
    best = 0
    for r, ch in enumerate(s):
        if ch in last and last[ch] >= l:
            # ch đã nằm trong window, di chuyển l qua sau vị trí cũ
            l = last[ch] + 1
        last[ch] = r
        best = max(best, r - l + 1)
    return best

# Ví dụ
print('Example 1:', longest_unique_substr('abcabcbb'))  # 3

# Edge cases: chuỗi rỗng, toàn ký tự giống nhau

# Mẹo phỏng vấn: nêu invariant và chứng minh rằng mỗi ký tự được xử lý tối đa một lần -> O(n).

# ----- Biến thể -----
# Nếu cần trả về chính substring (không chỉ độ dài), lưu lại vị trí bắt đầu tốt nhất.

# %%
# Cell 2: Two-sum (Hash map)
# ----- MỤC TIÊU -----
# Trong mảng nums, tìm hai chỉ số i,j sao cho nums[i]+nums[j] = target.
# ----- Ý TƯỞNG CHÍNH -----
# Lặp mảng, dùng dict để lưu giá trị đã thấy map[value] = index. Với mỗi x kiểm tra target-x có trong dict?
# ----- ĐỘ PHỨC TẠP -----
# Time: O(n), Space: O(n)

def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        need = target - x
        if need in seen:
            return [seen[need], i]
        seen[x] = i
    return None

print('Two-sum example:', two_sum([2,7,11,15], 9))

# Mẹo: xử lý duplicate bằng cách lưu index đầu tiên; nêu trade-off memory/time.

# %%
# Cell 3: 3-Sum (Sort + Two pointers)
# ----- MỤC TIÊU -----
# Tìm tất cả triplets (i<j<k) sao cho nums[i]+nums[j]+nums[k]=0 (unique triplets).
# ----- Ý TƯỞNG CHÍNH -----
# Sort mảng. Fix i, rồi dùng two pointers l=i+1, r=n-1 để tìm cặp có sum = -nums[i].
# Skip duplicates để tránh lặp kết quả.
# ----- ĐỘ PHỨC TẠP -----
# Time: O(n^2), Space: O(1) (ngoại trừ kết quả)

def three_sum(nums):
    nums.sort()
    n = len(nums)
    res = []
    for i in range(n-2):
        if i>0 and nums[i]==nums[i-1]:
            continue
        l, r = i+1, n-1
        while l<r:
            s = nums[i]+nums[l]+nums[r]
            if s==0:
                res.append([nums[i], nums[l], nums[r]])
                l+=1; r-=1
                while l<r and nums[l]==nums[l-1]: l+=1
                while l<r and nums[r]==nums[r+1]: r-=1
            elif s<0:
                l+=1
            else:
                r-=1
    return res

print('3-sum example:', three_sum([-1,0,1,2,-1,-4]))

# Mẹo: giải thích vì sao cần sort và cách skip duplicates.

# %%
# Cell 4: Binary search pattern (lower_bound)
# ----- MỤC TIÊU -----
# Tìm chỉ số nhỏ nhất i sao cho arr[i] >= target trong mảng đã sắp xếp.
# ----- Ý TƯỞNG CHÍNH -----
# Classic binary search trên khoảng [lo, hi). Giữ invariant: arr[lo..hi) là vùng candidate.
# ----- ĐỘ PHỨC TẠP -----
# Time: O(log n), Space: O(1)

def lower_bound(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo+hi)//2
        if arr[mid] < target:
            lo = mid+1
        else:
            hi = mid
    return lo

print('lower_bound example:', lower_bound([1,3,5,7], 4))

# Biến thể: tìm first True trong predicate monotonic (binary search on answer). Nên nêu ví dụ.

# %%
# Cell 5: Sliding window maximum (mono-deque)
# ----- MỤC TIÊU -----
# Với mảng nums và k, trả về max của mỗi subarray kích thước k.
# ----- Ý TƯỞNG CHÍNH -----
# Dùng deque lưu chỉ số với invariant: giá trị ở đuôi luôn giảm. Head chứa chỉ số của max.
# ----- ĐỘ PHỨC TẠP -----
# Time: O(n), Space: O(k)

from collections import deque

def max_sliding_window(nums, k):
    if not nums or k==0: return []
    dq = deque()
    res = []
    for i, x in enumerate(nums):
        # loại các index out of window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        # loại các value nhỏ hơn x
        while dq and nums[dq[-1]] < x:
            dq.pop()
        dq.append(i)
        if i >= k-1:
            res.append(nums[dq[0]])
    return res

print('Sliding window max example:', max_sliding_window([1,3,-1,-3,5,3,6,7], 3))

# Mẹo phỏng vấn: trình bày invariant và chứng minh O(n).

# %%
# Cell 6: Median of stream (two heaps)
# ----- MỤC TIÊU -----
# Thiết kế cấu trúc hỗ trợ insert số và truy vấn median trong streaming data.
# ----- Ý TƯỞNG CHÍNH -----
# Giữ hai heap: max-heap (lower half) và min-heap (upper half). Giữ kích thước cân bằng.
# ----- ĐỘ PHỨC TẠP -----
# Time per insert: O(log n), Space: O(n)

import heapq
class MedianFinder:
    def __init__(self):
        self.low = []  # max-heap via negatives
        self.high = []
    def addNum(self, num):
        if not self.low or num <= -self.low[0]:
            heapq.heappush(self.low, -num)
        else:
            heapq.heappush(self.high, num)
        if len(self.low) > len(self.high) + 1:
            heapq.heappush(self.high, -heapq.heappop(self.low))
        elif len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))
    def findMedian(self):
        if len(self.low) > len(self.high):
            return -self.low[0]
        return (-self.low[0] + self.high[0]) / 2.0

mf = MedianFinder()
for x in [5,2,3,4,1,6,7,0,8]:
    mf.addNum(x)
print('Median example:', mf.findMedian())

# Phỏng vấn: nêu invariant về sizes, chứng minh correctness.

# %%
# Cell 7: BFS - shortest path in grid (with obstacles)
# ----- MỤC TIÊU -----
# Tìm số bước ngắn nhất từ start -> goal trên grid 0/1 (0 free, 1 blocked).
# ----- Ý TƯỞNG CHÍNH -----
# Dùng BFS với queue, mỗi node gồm (r,c,dist). Mark visited khi enqueue.
# ----- ĐỘ PHỨC TẠP -----
# Time & Space: O(R*C)

from collections import deque

def shortest_path_grid(grid, start, goal):
    R, C = len(grid), len(grid[0])
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    q = deque([(start[0], start[1], 0)])
    seen = { (start[0], start[1]) }
    while q:
        r,c,d = q.popleft()
        if (r,c) == goal:
            return d
        for dr,dc in dirs:
            nr, nc = r+dr, c+dc
            if 0<=nr<R and 0<=nc<C and grid[nr][nc]==0 and (nr,nc) not in seen:
                seen.add((nr,nc))
                q.append((nr,nc,d+1))
    return -1

print('BFS grid example:', shortest_path_grid([[0,0,1],[0,0,0],[1,0,0]], (0,0), (2,2)))

# Edge: start==goal, or goal unreachable.

# %%
# Cell 8: DFS / Number of islands (connected components)
# ----- MỤC TIÊU -----
# Đếm số cụm '1' liên tiếp trong grid.
# ----- Ý TƯỞNG CHÍNH -----
# DFS hoặc BFS để mark toàn bộ cell trong 1 island.
# ----- ĐỘ PHỨC TẠP -----
# Time: O(R*C), Space: O(R*C) recursion stack

def num_islands(grid):
    if not grid: return 0
    R, C = len(grid), len(grid[0])
    seen = [[False]*C for _ in range(R)]
    def dfs(r,c):
        if r<0 or r>=R or c<0 or c>=C or seen[r][c] or grid[r][c]==0:
            return
        seen[r][c] = True
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            dfs(r+dr, c+dc)
    cnt = 0
    for i in range(R):
        for j in range(C):
            if grid[i][j]==1 and not seen[i][j]:
                cnt += 1
                dfs(i,j)
    return cnt

print('Islands example:', num_islands([[1,1,0],[0,1,0],[0,0,1]]))

# %%
# Cell 9: Union-Find (DSU) - template và ứng dụng
# ----- MỤC TIÊU -----
# Cấu trúc để gộp tập và truy vấn root trong thời gian gần O(1) (inverse Ackermann).
# ----- Ý TƯỞNG CHÍNH -----
# Path compression + union by rank/size.

class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        else:
            self.p[rb] = ra
            if self.r[ra] == self.r[rb]:
                self.r[ra] += 1
        return True

# Ứng dụng: number of connected components, detect cycle, island merging online

d = DSU(5)
d.union(0,1); d.union(1,2)
print('DSU find example:', d.find(2), d.find(3))

# Mẹo: nêu trade-offs vs BFS/DFS, khi DSU mạnh hơn (multiple union queries online).

# %%
# Cell 10: Topological sort (Kahn) - dependency resolution
# ----- MỤC TIÊU -----
# Trả về ordering các tasks theo dependency (DAG) hoặc [] nếu có cycle.
# ----- Ý TƯỞNG CHÍNH -----
# Dùng indegree array, push nodes indeg=0 vào queue, pop và giảm indeg neighbors.

from collections import deque

def topo_sort(n, edges):
    indeg = [0]*n
    g = [[] for _ in range(n)]
    for u,v in edges:
        g[u].append(v); indeg[v]+=1
    q = deque([i for i in range(n) if indeg[i]==0])
    res = []
    while q:
        u = q.popleft()
        res.append(u)
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v]==0:
                q.append(v)
    return res if len(res)==n else []

print('Topological example:', topo_sort(4, [(0,1),(0,2),(1,3),(2,3)]))

# Interview tip: discuss cycle detection and alternative DFS-based topo.

# %%
# Cell 11: Longest Increasing Subsequence (O(n log n))
# ----- MỤC TIÊU -----
# Tính độ dài dãy con tăng dài nhất.
# ----- Ý TƯỞNG CHÍNH -----
# Dùng array tails: tails[len-1] = smallest tail value of increasing subsequence length len.
# Với mỗi x, tìm vị trí bằng binary search và cập nhật tails.
# ----- ĐỘ PHỨC TẠP -----
# Time: O(n log n), Space: O(n)

import bisect

def lis_length(a):
    tails = []
    for x in a:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)

print('LIS example:', lis_length([10,9,2,5,3,7,101,18]))

# Mẹo: nếu cần reconstruct sequence, giữ parent pointer và index arrays.

# %%
# Cell 12: 0/1 Knapsack (1D DP optimization)
# ----- MỤC TIÊU -----
# Tối đa tổng value trong giới hạn weight W, đồ vật chỉ chọn 0/1.
# ----- Ý TƯỞNG CHÍNH -----
# DP[cap] = max value đạt được với capacity cap; duyệt items và cập nhật cap từ W xuống w.
# ----- ĐỘ PHỨC TẠP -----
# Time: O(n*W), Space: O(W)

def knapsack(weights, values, W):
    dp = [0]*(W+1)
    n = len(weights)
    for i in range(n):
        w, v = weights[i], values[i]
        for cap in range(W, w-1, -1):
            dp[cap] = max(dp[cap], dp[cap-w] + v)
    return dp[W]

print('Knapsack example:', knapsack([2,3,4,5],[3,4,5,6], 5))

# Mẹo: nêu trường hợp W quá lớn => dùng approximate/greedy hoặc meet-in-the-middle.

# %%
# Cell 13: KMP (prefix function) - string matching
# ----- MỤC TIÊU -----
# Tìm vị trí xuất hiện đầu tiên của pattern trong text (linear time)
# ----- Ý TƯỞNG CHÍNH -----
# Tính lps (longest proper prefix which is also suffix) cho pattern, dùng để né backtracking.

def kmp_search(text, pat):
    if not pat: return 0
    n, m = len(text), len(pat)
    lps = [0]*m
    length = 0
    i = 1
    while i < m:
        if pat[i] == pat[length]:
            length += 1; lps[i] = length; i += 1
        else:
            if length != 0:
                length = lps[length-1]
            else:
                lps[i] = 0; i += 1
    i = j = 0
    while i < n:
        if text[i] == pat[j]:
            i += 1; j += 1
            if j == m:
                return i - j
        else:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
    return -1

print('KMP example:', kmp_search('abxabcabcaby','abcaby'))

# Interview tip: nêu complexity O(n+m) và khi KMP hữu dụng.

# %%
# Cell 14: Greedy - Interval Scheduling (max non-overlapping intervals)
# ----- MỤC TIÊU -----
# Tối đa số interval không chồng chéo, sắp xếp theo end time
# ----- Ý TƯỞNG CHÍNH -----
# Greedy: chọn interval có end nhỏ nhất mỗi bước.

def max_non_overlapping(intervals):
    intervals.sort(key=lambda x: x[1])
    cnt = 0; last_end = -10**18
    for s,e in intervals:
        if s >= last_end:
            cnt += 1; last_end = e
    return cnt

print('Interval scheduling example:', max_non_overlapping([(1,3),(2,4),(3,5)]))

# Mẹo: đưa ra chứng minh ngắn rằng lựa chọn greedy là tối ưu.

# %%
# Cell 15: Backtracking - permutations & combinations
# ----- MỤC TIÊU -----
# Sinh tất cả hoán vị hoặc tổ hợp (combinations) của dãy nhỏ

def permutations(nums):
    res = []
    n = len(nums)
    used = [False]*n
    def dfs(cur):
        if len(cur) == n:
            res.append(cur.copy()); return
        for i in range(n):
            if used[i]: continue
            used[i] = True
            cur.append(nums[i])
            dfs(cur)
            cur.pop(); used[i] = False
    dfs([])
    return res

print('Permutations example:', permutations([1,2,3]))

def combinations(nums, k):
    res = []
    cur = []
    n = len(nums)
    def dfs(i):
        if len(cur) == k:
            res.append(cur.copy()); return
        if i >= n: return
        cur.append(nums[i]); dfs(i+1); cur.pop()
        dfs(i+1)
    dfs(0)
    return res

print('Combinations example:', combinations([1,2,3], 2))

# Tip: trình bày pruning nếu cần (ví dụ dùng remaining elements bound).

# %%
# Cell 16: Trie (prefix tree)
# ----- MỤC TIÊU -----
# Cài đặt cơ bản trie để hỗ trợ insert, search, startsWith

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True
    def search(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children: return False
            node = node.children[ch]
        return node.is_end
    def startsWith(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children: return False
            node = node.children[ch]
        return True

tr = Trie(); tr.insert('hello')
print('Trie example:', tr.search('hello'), tr.search('hell'), tr.startsWith('hell'))

# %%
# Cell 17: Dijkstra - weighted shortest path
# ----- MỤC TIÊU -----
# Tính khoảng cách ngắn nhất từ source tới mọi node với graph trọng số không âm

import math
import heapq

def dijkstra(n, adj, src):
    dist = [math.inf]*n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d,u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v,w in adj.get(u, []):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

adj = {0:[(1,4),(2,1)], 1:[(3,1)], 2:[(1,2),(3,5)], 3:[]}
print('Dijkstra example:', dijkstra(4, adj, 0))

# Interview tip: nêu complexity O((V+E) log V)

# %%
# Cell 18: Reservoir sampling
# ----- MỤC TIÊU -----
# Sample k items uniformly from streaming iterator of unknown length

def reservoir_sample(stream, k):
    res = []
    for i, x in enumerate(stream):
        if i < k:
            res.append(x)
        else:
            j = random.randrange(i+1)
            if j < k:
                res[j] = x
    return res

print('Reservoir sampling example:', reservoir_sample(range(100), 5))

# Mẹo: dùng khi dataset quá lớn để load full vào RAM.

# %%
# Cell 19: Practice harness - tự động test một số hàm mẫu
PRACTICE = [
    ('Longest unique substring', lambda: longest_unique_substr('pwwkew')),
    ('Two-sum', lambda: two_sum([2,7,11,15], 9)),
    ('LIS length', lambda: lis_length([10,9,2,5,3,7,101,18])),
]

def test_harness():
    print('Running quick tests...')
    for name, fn in PRACTICE:
        try:
            print(name, '->', fn())
        except Exception as e:
            print(name, 'raised error', e)

if __name__ == '__main__':
    test_harness()

# %%
# Cell 20: Tổng kết - checklist phỏng vấn
# 1) Nêu ý tưởng & invariant trước khi code.
# 2) Viết pseudo-code ngắn nếu cần.
# 3) Test với vài edge cases.
# 4) Phân tích complexity và nói cách tối ưu nếu thời gian.
# 5) Nếu interviewer muốn tối ưu thêm, nói rõ trade-offs.

# KẾT THÚC
# Nếu bạn muốn: mình sẽ
# - A) chuyển thành .ipynb (Jupyter) với markdown đẹp hơn cho mỗi cell,
# - B) bổ sung 20 bài tập + giải chi tiết từng bước,
# - C) tổ chức mock-interview (mình hỏi, bạn code, mình phản hồi).
# Chọn A/B/C nhé.
