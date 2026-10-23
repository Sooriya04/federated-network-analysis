# Dataset Information and Verification Document (22ITPA0)

This document provides verified provenance, structural semantics, topological parameters, and bibliographic citations for the datasets utilized in this project. All numerical statistics are directly computed from the raw data files loaded via the preprocessing pipeline.

---

## 1. Primary Dataset: Fedivertex (PeerTube Follow Network)

* **Dataset Identifier:** `peertube`
* **Subnetwork / Snapshot:** PeerTube Follow Federation Network (Snapshot: `2026-04-13`)
* **Source Repository / URL:** [arXiv:2505.20882](https://arxiv.org/abs/2505.20882) and [Kaggle Dataset](https://www.kaggle.com/datasets/marcdamie/fedivertex)
* **Authors / Reference:** Marc Damie and Edwige Cyffers, *"Fedivertex: a Graph Dataset based on Decentralized Social Networks for Trustworthy Machine Learning"*, Proceedings of the ACM Web Conference 2026 (WWW '26), May 2026.
* **Publication Year:** 2026 (arXiv preprint deposited May 2025; published at WWW '26).
* **Access Date:** 2026-10-04
* **Network Type:** DIRECTED graph ($G = (V, E)$)
* **Node Meaning:** An individual PeerTube instance (an autonomous, federated video hosting and streaming server within the Fediverse decentralized ecosystem, e.g., `framatube.org`, `makertube.net`, `video.blender.org`).
* **Edge Meaning:** A directed follow / federation relationship ($u \to v$). Instance $u$ follows or subscribes to instance $v$, enabling automated syndication, indexing, and relaying of video channels published on $v$ to users on $u$.
* **Edge Weights:** Integer weight representing interaction/channel subscription density (unweighted topological projection used for structural invariants).

### Computed Graph Statistics (from Loaded Raw Files)
* **Raw Edge Records:** 43,888
* **Raw Unique Nodes:** 1,553
* **Null / Missing Value Rows:** 0 (Verified: no null or corrupt entries)
* **Self-Loops Detected & Removed:** 9 ($u = v$)
* **Duplicate Directed Edges Removed:** 0
* **Cleaned Directed Graph ($G$):**
  * **Total Nodes ($N$):** 1,553
  * **Total Directed Edges ($M$):** 43,879
  * **Average Degree ($\langle k_{in} \rangle = \langle k_{out} \rangle$):** 28.2543
  * **Graph Density:** 0.0182
* **Component Decomposition:**
  * **Weakly Connected Components (WCC):** 4
    * **Largest WCC ($G_{WCC}$):** 1,547 nodes (99.61% of all nodes), 43,876 edges (99.99% of all edges)
  * **Strongly Connected Components (SCC):** 986
    * **Largest SCC ($G_{SCC}$):** 534 nodes (34.39% of all nodes), 18,718 edges (42.66% of all edges)
* **Items Marked TO VERIFY:** None. All parameters, citations, and counts were directly verified from the paper and downloaded dataset archives.

---

## 2. Secondary / Alternative Dataset: SNAP wiki-Vote

* **Dataset Identifier:** `wiki_vote`
* **Source Repository / URL:** [https://snap.stanford.edu/data/wiki-Vote.html](https://snap.stanford.edu/data/wiki-Vote.html)
* **Authors / Reference:** Jure Leskovec, Daniel Huttenlocher, Jon Kleinberg, *"Predicting Positive and Negative Links in Online Social Networks"*, Proceedings of the 19th International Conference on World Wide Web (WWW '10), pp. 641–650, 2010.
* **Publication Year:** 2010 (Data collection window: Wikipedia inception in 2001 through January 2008).
* **Access Date:** 2026-10-04
* **Network Type:** DIRECTED graph ($G = (V, E)$)
* **Node Meaning:** Wikipedia user accounts who participated as voters or candidates in Requests for Adminship (RfA) elections.
* **Edge Meaning:** A directed vote from user $i$ to candidate $j$ ($i \to j$) cast during an RfA election.
* **Edge Weights:** Unweighted.

### Computed Graph Statistics (from Loaded Raw Files)
* **Raw Edge Records:** 103,689
* **Raw Unique Nodes:** 7,115
* **Null / Missing Value Rows:** 0
* **Self-Loops Detected & Removed:** 0
* **Duplicate Directed Edges Removed:** 0
* **Cleaned Directed Graph ($G$):**
  * **Total Nodes ($N$):** 7,115
  * **Total Directed Edges ($M$):** 103,689
  * **Average Degree:** 14.5733
  * **Graph Density:** 0.0020
* **Component Decomposition:**
  * **Weakly Connected Components (WCC):** 24
    * **Largest WCC ($G_{WCC}$):** 7,066 nodes (99.31%), 103,663 edges (99.97%)
  * **Strongly Connected Components (SCC):** 5,816
    * **Largest SCC ($G_{SCC}$):** 1,300 nodes (18.27%), 39,456 edges (38.05%)
* **Items Marked TO VERIFY:** None. Verified against official Stanford SNAP repository.

---

## 3. Methodological Justification for Component Selection Across Tasks

| Analysis Task | Target Subgraph | Mathematical & Substantive Justification |
| :--- | :--- | :--- |
| **In/Out Degree Centrality** | Full Cleaned Graph ($G$) | Preserves complete nominal counts of all followers and nominations without boundary truncation. |
| **PageRank ($d=0.85, 0.50, 0.99$)** | Full Cleaned Graph ($G$) | The random surfer model naturally accommodates dangling sink nodes via uniform teleportation without graph pruning. |
| **Kleinberg HITS (Hubs & Authorities)** | Full Cleaned Graph ($G$) | Captures pure authorities (sink nodes with $k_{out}=0$) and pure hubs ($k_{in}=0$), which are critical for detecting structural asymmetry. |
| **Closeness & Betweenness Centrality** | Largest WCC ($G_{WCC}$) | Prevents infinite shortest path lengths ($\infty$) and division-by-zero errors that occur across disconnected components. |
| **Prestige (In-Degree, Proximity, Katz)** | Full Graph ($G$) / Directed Reachability | Formulated with reachability influence domains $I(i)$, where unreachable nodes contribute infinite distance or zero rank weight. |
| **Small-World ($\sigma, C, L$) & BA Model** | Undirected LCC ($G_{LCC}$) | Characteristic shortest path length $L$ and clustering $C$ require a single connected topological space for valid geodesic comparisons. |
| **Link Prediction (CN, JC, AA, PA, RA)** | Undirected LCC ($G_{LCC}$) | Guarantees that negative sampling occurs within the reachable component and training graph remains connected via spanning tree. |
| **Granovetter Threshold Cascades** | Largest WCC ($G_{WCC}$) | Ensures cascade dynamics reflect topological complex contagion rather than artificial termination at disconnected singleton boundaries. |
| **Random Walk & Restart Convergence** | Full Cleaned Graph ($G$) | Direct validation of the Ergodic Theorem against the global PageRank stationary vector. |
