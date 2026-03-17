"""
30 Python function signatures for perturbation experiments.

These are separate from the matched pairs in programming_prompts.py,
so they can be used as source activations while the pairs are used
for computing the Python↔Haskell DoM direction.
"""

PYTHON_PROMPTS_HOLDOUT = [
    "def count_vowels(s):",
    "def rotate_list(arr, k):",
    "def is_anagram(s1, s2):",
    "def transpose_matrix(matrix):",
    "def longest_common_prefix(strs):",
    "def merge_intervals(intervals):",
    "def group_by_key(pairs):",
    "def moving_average(data, window):",
    "def deep_copy(obj):",
    "def to_roman(num):",
    "def run_length_encode(s):",
    "def chunk_list(arr, size):",
    "def invert_dict(d):",
    "def nth_prime(n):",
    "def levenshtein(s1, s2):",
    "def topological_sort(graph):",
    "def sliding_window_max(nums, k):",
    "def serialize_tree(root):",
    "def lru_cache(capacity):",
    "def find_median(nums):",
    "def permutations(arr):",
    "def knapsack(weights, values, capacity):",
    "def dijkstra(graph, start):",
    "def huffman_encode(text):",
    "def balanced_parens(n):",
    "def longest_increasing_subsequence(arr):",
    "def convex_hull(points):",
    "def polynomial_multiply(p1, p2):",
    "def reservoir_sample(stream, k):",
    "def strongly_connected(graph):",
]
