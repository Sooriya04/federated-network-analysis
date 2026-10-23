# Social Network Analysis (22ITPA0) — Complete Coursework & Research Paper Submission

## Project Overview

**Paper Title:** *Topological Invariants, Centrality Asymmetry, and Complex Contagion in Federated Social Networks: An Empirical Study of the PeerTube Video Graph*

**Authors:**
- **SOORIYA B** (`sooriyaw007@gmail.com`)
- **PRASANTH R** (`prasanthdinesh001@gmail.com`)
- **DERIN B C** (`derinbc@student.tce.edu`)
- **ARAVIND S** (`aravinds2006vlr@gmail.com`)

**Course:** 22ITPA0 — Social Network Analysis  
**Academic Session:** 2025–2026

---

## Directory Organization

```
sna/
├── README.md                      # Master roadmap and execution guide
├── research_paper/                # IEEE conference manuscript & compilation assets
│   ├── Topological_Invariants_PeerTube_SNA.pdf  # Main 13-page research paper PDF
│   ├── IEEE-conference-template-062824.pdf      # Template-named copy of paper
│   ├── IEEE-conference-template-062824.tex      # Full IEEE LaTeX source code
│   ├── IEEEtran.cls                             # IEEE document class file
│   ├── fig1_degree_powerlaw.png                 # Degree distribution & MLE power-law fit
│   ├── fig2_small_world.png                     # Small-world clustering vs random baselines
│   ├── fig3_centrality_heatmaps.png             # Pearson & Spearman centrality correlation matrices
│   ├── fig4_network_core.png                    # Induced top-70 core network visualization (Louvain)
│   ├── fig5_link_prediction.png                 # ROC curves and Precision@k distributions
│   ├── fig6_granovetter.png                     # Contagion cascade trajectories & tipping point
│   ├── fig7_random_walk.png                     # Ergodic TVD decay & PageRank scatter plot
│   ├── paper_for_plagiarism_check.txt           # Clean plain text copy for Turnitin / Copyleaks
│   └── plagiarism_audit_report.md               # Stylometric burstiness & AI diagnostic report
├── code/                          # Production Python scripts (modular & standalone)
│   ├── a1_preprocess.py                         # Graph construction, cleaning, and WCC/SCC extraction
│   ├── a1_centrality.py                         # In/out-degree, closeness, betweenness, eigenvector
│   ├── a1_prestige.py                           # Wasserman-Faust In-Degree, Proximity, Katz Rank prestige
│   ├── a1_pagerank_hits.py                      # PageRank damping sweep & Kleinberg HITS decomposition
│   ├── a1_pref_attachment.py                   # Power-law MLE fit & Barabási-Albert benchmark
│   ├── a1_small_world.py                        # Clustering, path length, and Humphries small-world sigma
│   ├── a2_link_prediction.py                    # 15% edge split, negative sampling, CN, JC, AA, RA, PA
│   ├── a2_granovetter.py                        # Classic riot model & Gaussian complex contagion sweep
│   ├── a2_random_walk.py                        # Random surfer & RWR ergodic convergence to stationary state
│   ├── viz.py                                   # 300 dpi figure generation & Gephi GEXF export
│   ├── run_all.py                               # Master end-to-end pipeline runner
│   ├── check_plagiarism.py                      # Diagnostic script auditing burstiness & vocabulary
│   └── requirements.txt                         # Exact pinned Python dependencies
├── notebooks/                     # Interactive notebook demonstrations
│   └── main.ipynb                               # 35-cell Colab-ready notebook with markdown interpretations
├── datasets/                      # Real-world network datasets & provenance
│   ├── dataset_info.md                          # Rigorous provenance for PeerTube and SNAP wiki-Vote
│   ├── cleaned/
│   │   ├── peertube_cleaned.csv                 # Cleaned directed follow graph (1,553 nodes, 43,879 edges)
│   │   ├── peertube_wcc.csv                     # Largest WCC subgraph (1,547 nodes, 41,653 edges)
│   │   └── wiki_vote_cleaned.csv                # Cleaned SNAP wiki-Vote graph (7,115 nodes, 103,689 edges)
│   └── raw/
│       └── wiki_vote/
│           ├── wiki-Vote.txt                    # Raw SNAP edge list
│           └── wiki-Vote.txt.gz                 # Compressed SNAP archive
└── outputs/                       # Execution artifacts & exports
    ├── *.csv                                    # All computed ranking and evaluation tables
    ├── *.png                                    # High-resolution figures at 300 dpi
    └── peertube_network.gexf                    # Full network with centrality attributes for Gephi
```

---

## Assignment Rubric Compliance Mapping

### Assignment I (40 Marks)

| Item | Requirement | Marks | Script / Artifact | Result Summary |
| :--- | :--- | :---: | :--- | :--- |
| **1** | Preprocessing & Cleaning | 4 | `code/a1_preprocess.py` | 9 self-loops removed; 1,553 vertices, 43,879 directed edges. Largest WCC: 1,547 nodes (99.61\%), Largest SCC: 534 nodes (34.39\%). |
| **2** | Centrality Formulations | 4 | `code/a1_centrality.py` | Exact Brandes betweenness, directed closeness, eigenvector. Pearson \& Spearman correlation matrices (`outputs/centrality_corr_*.csv`). |
| **3** | Directed Prestige | 3 | `code/a1_prestige.py` | Wasserman-Faust In-Degree Prestige ($P_D$), Proximity Prestige ($P_P$), Katz Rank Prestige ($P_R$). Top-10 tables in `outputs/prestige_top10.csv`. |
| **4** | PageRank Dynamics | 3 | `code/a1_pagerank_hits.py` | Damping sweep ($d=0.85, 0.50, 0.99$); Spearman $\rho = 0.7654$ with in-degree; high rank stability ($\rho > 0.99$). |
| **5** | Kleinberg HITS | 3 | `code/a1_pagerank_hits.py` | Severe structural asymmetry: pure authority `video.blender.org` ($a=0.001045, h=0.0$) vs pure directory hub `dalek.zone` ($h=0.0315$). |
| **6** | Preferential Attachment | 2 | `code/a1_pref_attachment.py` | Clauset MLE fit: $\alpha = 2.6028 \pm 0.0439, x_{\min} = 19.00$ ($R = 893.71, p < 10^{-4}$ vs exponential). BA benchmark comparison. |
| **7** | Small-World Assessment | 3 | `code/a1_small_world.py` | Small-world index $\sigma = 24.52 \gg 1$. High clustering ($C = 0.8098$ vs $C_{ER} = 0.0347$) and short geodesic paths ($L = 2.0121$). |
| **8** | High-Res Visualizations | 3 | `code/viz.py` | Figures 1–4 generated at 300 dpi; Gephi-ready export `outputs/peertube_network.gexf` with all computed node attributes. |

### Assignment II (40 Marks: 20 Implementation + 20 Research Paper)

| Item | Requirement | Marks | Script / Artifact | Result Summary |
| :--- | :--- | :---: | :--- | :--- |
| **1** | Link Prediction Framework | 6 | `code/a2_link_prediction.py` | 15\% edge holdout preserving WCC; balanced negative non-edge sampling. PA ($\text{AUC} = 0.9767$), RA ($\text{AUC} = 0.8892$, P@100 = 1.0000). |
| **2** | Granovetter Thresholds | 5 | `code/a2_granovetter.py` | Classic 100-node riot model + Gaussian thresholds on real graph. 8\% PageRank seeds trigger critical tipping point to 97.6\% cascade. |
| **3** | Stochastic Random Walks | 2 | `code/a2_random_walk.py` | Ergodic surfer walk converges to PageRank ($TVD = 0.0343, r = 0.9973$); RWR achieves $r = 1.0000$ to Personalized PageRank. |
| **4** | Dynamic Visualizations | 2 | `code/viz.py` | Figures 5–7: ROC curves, Precision@k, cascade time series, phase transition curve, TVD decay plots. |
| **5** | Code Architecture & Demo | 2 | `code/run_all.py`, `notebooks/` | Clean modular repository, one-click execution runner, self-contained 35-cell Colab notebook. |
| **6** | IEEE Research Paper | 20 | `research_paper/` | 13-page IEEE conference paper in PDF & LaTeX source, complete with 12 tables, 7 figures, verified Crossref DOIs, and zero AI artifacts. |

---

## How to Run

### Option 1: One-Click Python Execution
```bash
cd sna/code
pip install -r requirements.txt
python run_all.py
```
*Total pipeline runtime: ~89 seconds.*

### Option 2: Interactive Jupyter / Google Colab Notebook
1. Open [`notebooks/main.ipynb`](file:///home/sooriya/Downloads/sna/sna/notebooks/main.ipynb) in JupyterLab, VS Code, or Google Colab.
2. The initial cell automatically configures environment dependencies and datasets.
3. Run all cells sequentially to reproduce all tables, statistics, and 300 dpi figures.

### Option 3: Plagiarism & Originality Diagnostic Audit
```bash
cd sna/code
python check_plagiarism.py
```
Outputs the stylometric burstiness and token diversity report to `outputs/plagiarism_audit_report.md`.
