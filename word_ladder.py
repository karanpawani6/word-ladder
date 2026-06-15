import streamlit as st
import time
import sys
sys.setrecursionlimit(10000)
import numpy as np
from collections import deque
import heapq

@st.cache_resource
def load_data():
    words = []
    vectors = []
    with open('glove.100d.20000.txt', 'r', encoding='utf-8') as glove:
        for line in glove:
            values = line.split()
            words.append(values[0])
            vectors.append(list(map(float, values[1:])))

    vectors = np.array(vectors, dtype=np.float32)
    words = np.array(words)

    # Normalize all vectors once — so dot product = cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

    word2index = {word: i for i, word in enumerate(words)}

    return words, word2index, vectors

words, word2index, vectors = load_data()

def get_similarity(word1, word2):
    if word1 in word2index and word2 in word2index:
        i = word2index[word1]
        j = word2index[word2]
        # dot product of normalized vectors = cosine similarity
        return float(np.dot(vectors[i], vectors[j]))
    return None

def k_neighbors(word, k):
    if word not in word2index:
        return []
    k = max(5, min(50, k))
    idx = word2index[word]
    # dot product with all words at once = cosine similarity (vectors are normalized)
    sims = vectors @ vectors[idx]
    # exclude the word itself
    sims[idx] = -999
    top_k = np.argpartition(sims, -k)[-k:]
    # sort them by similarity (best first)
    top_k = top_k[np.argsort(-sims[top_k])]
    return words[top_k]

#---------------------------------- BFS ------------------------------------------

def BFS(start_W, goal_W, k):
    if start_W not in word2index or goal_W not in word2index:
        return None, 0
    if start_W == goal_W:
        return [start_W], 1
    frontier = deque()
    frontier.append((start_W, [start_W]))
    explored = set()
    node_expanded = 0
    while frontier:
        node, path = frontier.popleft()
        node_expanded += 1
        explored.add(node)
        for child in k_neighbors(node, k):
            if child not in explored:
                new_path = path + [child]
                if child == goal_W:
                    return new_path, int(node_expanded)
                frontier.append((child, new_path))
    return None, node_expanded

#----------------------------- DFS --------------------------------------

def recursive_dls(node, goal_W, limit, path, explored, node_expanded, k):
    node_expanded[0] += 1
    explored.add(node)
    if node == goal_W:
        return path, "success"
    elif limit == 0:
        return None, "cutoff"
    else:
        cutoff_occurred = False
        for child in k_neighbors(node, k):
            if child not in explored:
                result_path, result = recursive_dls(
                    child, goal_W, limit - 1,
                    path + [child], explored, node_expanded, k
                )
                if result == "cutoff":
                    cutoff_occurred = True
                elif result != "failure":
                    return result_path, "success"
        if cutoff_occurred:
            return None, "cutoff"
        else:
            return None, "failure"

def DFS(start_W, goal_W, limit, k):
    if start_W not in word2index or goal_W not in word2index:
        return None, 0
    explored = set()
    node_expanded = [0]
    path, result = recursive_dls(
        start_W, goal_W, limit, [start_W], explored, node_expanded, k
    )
    return path, node_expanded[0]

#----------------------------- UCS --------------------------------------

def UCS(start_W, goal_W, k):
    if start_W not in word2index or goal_W not in word2index:
        return None, 0
    pq = []
    heapq.heappush(pq, (0.0, start_W))
    explored = set()
    parent = {start_W: None}
    node_expanded = 0
    while pq:
        cost, node = heapq.heappop(pq)
        if node == goal_W:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1], node_expanded
        if node in explored:
            continue
        explored.add(node)
        node_expanded += 1
        for child in k_neighbors(node, k):
            if child not in explored:
                edge_cost = 1 - (get_similarity(node, child) or 0)
                new_cost = cost + edge_cost
                if child not in parent:
                    parent[child] = node
                heapq.heappush(pq, (new_cost, child))
    return None, node_expanded

#----------------------------- Greedy --------------------------------------

def Greedy(start_W, goal_W, k):
    if start_W not in word2index or goal_W not in word2index:
        return None, 0
    pq = []
    h = 1 - (get_similarity(start_W, goal_W) or 0)
    heapq.heappush(pq, (h, start_W))
    explored = set()
    parent = {start_W: None}
    node_expanded = 0
    while pq:
        _, node = heapq.heappop(pq)
        if node == goal_W:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1], node_expanded
        if node in explored:
            continue
        explored.add(node)
        node_expanded += 1
        for child in k_neighbors(node, k):
            if child not in explored:
                h = 1 - (get_similarity(child, goal_W) or 0)
                if child not in parent:
                    parent[child] = node
                heapq.heappush(pq, (h, child))
    return None, node_expanded

#----------------------------- A* --------------------------------------

def Astar(start_W, goal_W, k):
    if start_W not in word2index or goal_W not in word2index:
        return None, 0
    pq = []
    h = 1 - (get_similarity(start_W, goal_W) or 0)
    heapq.heappush(pq, (h, start_W, 0.0))
    explored = set()
    parent = {start_W: None}
    best_g = {start_W: 0.0}
    node_expanded = 0
    while pq:
        f, node, g = heapq.heappop(pq)
        if node == goal_W:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1], node_expanded
        if node in explored:
            continue
        explored.add(node)
        node_expanded += 1
        for child in k_neighbors(node, k):
            if child not in explored:
                edge_cost = 1 - (get_similarity(node, child) or 0)
                new_g = g + edge_cost
                new_h = 1 - (get_similarity(child, goal_W) or 0)
                new_f = new_g + new_h
                if new_g < best_g.get(child, float('inf')):
                    best_g[child] = new_g
                    parent[child] = node
                    heapq.heappush(pq, (new_f, child, new_g))
    return None, node_expanded

#----------------------------- GUI --------------------------------------

st.title("Word Ladder Search in Semantic Embedding Space")

start_word = st.text_input("Enter Start Word")
goal_word  = st.text_input("Enter Goal Word")
algorithm  = st.selectbox("Select Algorithm", ["BFS", "DFS", "UCS", "Greedy", "A*"])
k          = st.slider("Select k (neighbors)", min_value=5, max_value=50, value=10)

if st.button("Search"):
    if start_word not in word2index:
        st.error(f"'{start_word}' not in vocabulary!")
    elif goal_word not in word2index:
        st.error(f"'{goal_word}' not in vocabulary!")
    else:
        with st.spinner("Searching..."):
            start_time = time.time()

            if algorithm == "BFS":
                path, nodes = BFS(start_word, goal_word, k)
            elif algorithm == "DFS":
                path, nodes = DFS(start_word, goal_word, 4000, k)
            elif algorithm == "UCS":
                path, nodes = UCS(start_word, goal_word, k)
            elif algorithm == "Greedy":
                path, nodes = Greedy(start_word, goal_word, k)
            elif algorithm == "A*":
                path, nodes = Astar(start_word, goal_word, k)

            runtime = time.time() - start_time

        if path:
            st.success("Path Found!")
            st.write("**Word Ladder:**", " → ".join(path))
            st.write("**Steps:**", len(path) - 1)
            st.write("**Nodes Expanded:**", nodes)
            st.write("**Time:**", f"{runtime:.4f} seconds")
        else:
            st.error("No path found!")
            st.write("**Nodes Expanded:**", nodes)
            st.write("**Time:**", f"{runtime:.4f} seconds")