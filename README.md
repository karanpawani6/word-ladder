# Word Ladder Search in Semantic Embedding Space

> Assignment #1 — Introduction to Artificial Intelligence  
> Institute of Business Administration (IBA), Karachi — Spring 2026  
> Instructor: Dr. Syed Ali Raza

---

## What is this project?

In the classic Word Ladder puzzle, you transform one word into another by changing one letter at a time. This project generalizes that idea using **word embeddings**.

Instead of changing letters, we move through **semantic space** — a mathematical space where words with similar meanings are placed close to each other. Each word is represented as a 100-dimensional vector (a list of 100 numbers) learned from millions of sentences using **GloVe (Global Vectors for Word Representation)**.

The goal: find a path from a **start word** to a **goal word** by stepping through semantically similar words, using different AI search algorithms.

---

## Live Demo

🔗 **[Try it here](https://your-deployment-link.streamlit.app)**  
*(No installation needed — runs in your browser)*

---

## How it works

Each word is a **state**. From any word, you can move to its **k most semantically similar neighbors** (k is adjustable from 5 to 50). The search algorithms explore this implicit graph to find a path from the start word to the goal word.

**Similarity** between two words is measured using **cosine similarity**:

```
CosSim(a, b) = (a · b) / (||a|| × ||b||)
```

The **edge cost** between two words is:

```
cost(u, v) = 1 - CosSim(u, v)
```

So moving between very similar words costs nearly 0, and moving between unrelated words costs nearly 2.

---

## Search Algorithms Implemented

| Algorithm | Type | Optimality | Speed |
|-----------|------|------------|-------|
| BFS | Uninformed | ✅ Shortest hops | Slow for distant pairs |
| DFS | Uninformed | ❌ No guarantee | Fast but poor quality |
| UCS | Uninformed | ✅ Lowest cost path | Moderate |
| Greedy Best-First | Informed | ❌ Not optimal | Very fast |
| A* | Informed | ✅ Optimal + efficient | Best overall |

### Heuristic Function (used by Greedy and A*)

```
h(s) = 1 - CosSim(s, goal)
```

This estimates how far the current word is from the goal using their direct cosine distance. Words already close to the goal get a low heuristic value and are explored first.

---

## Key Findings from Experiments

Six word pairs were tested across three values of k ∈ {5, 15, 50}. Here are the main takeaways:

### A* is the best overall algorithm
For `timer → worker` at k=5:
- BFS expanded **3,747 nodes** in 2.38 seconds
- A* expanded only **258 nodes** in 0.17 seconds
- Same path length (7 steps) — A* was ~14× more efficient

### Greedy is fastest but not optimal
For `Apanah → casasola` at k=5:
- BFS found the path in **6 steps**
- Greedy found a path in **11 steps** (nearly double)
- Greedy rushes toward the goal without considering total path cost

### DFS is unsuitable for this problem
DFS consistently hit the depth limit (4000) and returned paths with **3000–4000 steps** — practically useless compared to BFS finding the same pair in 7 steps.

### k is critical for reachability
The pair `timer → panah` had **no path at k=5** (graph too disconnected). At k=15, all algorithms found a path in 6 steps. At k=50, path length dropped to 4 steps. A small k can make many word pairs completely unreachable.

### Larger k = shorter paths (for most algorithms)
For `scarce → goyard`:
- k=5: BFS found path in **12 steps**, expanded 61,564 nodes
- k=50: BFS found path in **4 steps**, expanded only 15,209 nodes

---

## Project Structure

```
word-ladder/
├── word_ladder.py          # Main application (all algorithms + Streamlit GUI)
├── glove.100d.20000.txt    # GloVe embeddings (20,000 words, 100 dimensions)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/word-ladder.git
cd word-ladder
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run word_ladder.py
```

---

## Dependencies

```
streamlit
scikit-learn
numpy
```

---

## About GloVe Embeddings

GloVe (Global Vectors for Word Representation) is a pretrained word embedding model developed at Stanford. It learns word vectors by analyzing how often words co-occur across a large text corpus. Words used in similar contexts end up with similar vectors — which is what makes semantic search possible here.

This project uses the **100-dimensional, 20,000-word** restricted vocabulary version.

🔗 [Learn more about GloVe](https://nlp.stanford.edu/projects/glove/)

---

## Course Context

This assignment was completed as part of the **Introduction to Artificial Intelligence** course at IBA Karachi (Spring 2026). It covers:

- State-space search problem formulation
- Uninformed search (BFS, DFS, UCS)
- Informed search with heuristics (Greedy, A*)
- Empirical algorithm evaluation
- Heuristic admissibility and consistency analysis
