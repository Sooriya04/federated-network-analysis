# Federated Network Analysis

An open-source empirical research and analysis toolkit investigating topological invariants, directional prestige asymmetry, link predictability, and complex contagion dynamics in decentralized federated social graphs (ActivityPub / PeerTube).

This repository provides the complete experimental pipeline, sanitized instance-level graph datasets, interactive Jupyter / Google Colab notebooks, Gephi Lite browser visualization files, and the full IEEE-formatted research paper.

---

## 👥 Authors & Contributors

* **SOORIYA B** — [sooriyaw007@gmail.com](mailto:sooriyaw007@gmail.com)
* **PRASANTH R** — [prasanthdinesh001@gmail.com](mailto:prasanthdinesh001@gmail.com)
* **DERIN B C** — [derinbc@student.tce.edu](mailto:derinbc@student.tce.edu)
* **ARAVIND S** — [aravinds2006vlr@gmail.com](mailto:aravinds2006vlr@gmail.com)

*Department of Information Technology, Thiagarajar College of Engineering, Madurai, Tamil Nadu, India.*

---

## 📄 Research Paper

The complete research paper is located in [`research_paper/`](research_paper/):

* **Title**: *Topological Invariants, Centrality Asymmetry, and Complex Contagion in Federated Social Networks: A Directed Graph Structural Analysis*
* **Compiled Manuscript**: [`research_paper/IEEE-conference-template-062824.pdf`](research_paper/IEEE-conference-template-062824.pdf) (13-page double-column IEEE format)
* **LaTeX Source**: [`research_paper/IEEE-conference-template-062824.tex`](research_paper/IEEE-conference-template-062824.tex)
* **Document Class**: [`research_paper/IEEEtran.cls`](research_paper/IEEEtran.cls)

---

## 📁 Repository Structure

```text
federated-network-analysis/
├── research_paper/       # Full research paper (PDF, LaTeX source, figures)
│   ├── IEEE-conference-template-062824.pdf
│   ├── IEEE-conference-template-062824.tex
│   ├── IEEEtran.cls
│   └── fig1..7_*.png    # High-resolution publication figures
├── code/                 # Modular Python analysis scripts
│   ├── a1_preprocess.py         # Graph sanitization & component extraction
│   ├── a1_centrality.py         # Degree, Closeness, Betweenness, Eigenvector
│   ├── a1_prestige.py           # Wasserman-Faust In-Degree, Proximity, Katz
│   ├── a1_pagerank_hits.py      # PageRank damping sweep & Kleinberg HITS
│   ├── a1_small_world.py        # Small-world sigma vs ER & rewiring
│   ├── a1_pref_attachment.py   # Clauset MLE power-law fit
│   ├── a2_link_prediction.py    # CN, Jaccard, AA, RA, PA benchmark
│   ├── a2_granovetter.py        # Complex contagion threshold diffusion
│   ├── a2_random_walk.py        # Markov chain TVD convergence
│   ├── export_gephi_lite.py     # Gephi Lite coordinates & layout exporter
│   ├── config.py / viz.py       # Configuration & visualization styling
│   ├── run_all.py               # Master end-to-end pipeline runner
│   └── requirements.txt         # Dependencies manifest
├── datasets/             # Sanitized network datasets & provenance
│   ├── cleaned/peertube_cleaned_edges.csv  # Cleaned directed follow graph (1,553 nodes, 43,879 links)
│   └── dataset_info.md                     # Provenance and attribute schema
├── gephi_lite/           # Instant web-based graph visualization assets
│   ├── peertube_gephi_lite.json  # Graphology JSON (instant drag-and-drop)
│   ├── peertube_gephi_lite.gexf  # GEXF 1.2 with viz coordinates & colors
│   ├── gephi_nodes.csv           # Node table with all centrality metrics
│   └── gephi_edges.csv           # Directed edge table
├── notebooks/            # Interactive exploration & Google Colab
│   ├── main.ipynb               # Step-by-step Jupyter notebook
│   └── single_cell_colab.py     # All-in-one Google Colab execution script
└── outputs/              # Exported publication figures & numerical result tables
```

---

## 🚀 Quick Start

### 1. Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/Sooriya04/federated-network-analysis.git
cd federated-network-analysis
pip install -r code/requirements.txt
```

### 2. Run the Complete Pipeline

To reproduce all 11 analytical stages and generate all output tables and figures:

```bash
python code/run_all.py
```

### 3. Interactive Notebook / Google Colab

* Run locally:
  ```bash
  jupyter notebook notebooks/main.ipynb
  ```
* Or copy the contents of `notebooks/single_cell_colab.py` into a blank cell on [Google Colab](https://colab.research.google.com/) for instant GPU/CPU cloud execution without local setup.

---

## 🌐 Instant Web Visualization with Gephi Lite

No desktop software installation required:

1. Open **[gephi.org/gephi-lite](https://gephi.org/gephi-lite/)** in your web browser.
2. Drag and drop **`gephi_lite/peertube_gephi_lite.json`** into the browser window.
3. The graph will render immediately with pre-computed ForceAtlas2 coordinates, Louvain modular community colors, and PageRank proportional node sizing.

---

## 📊 Summary of Empirical Findings

| Metric / Experiment | Value / Result | Significance |
| :--- | :--- | :--- |
| **Network Scale** | 1,553 nodes, 43,879 edges | Complete sanitized PeerTube federated graph |
| **Reciprocity** | 28.59% | Over 71% of syndication ties are strictly unidirectional |
| **Degree Scaling Exponent** | $\alpha = 2.6028$ ($x_{\min}=15$) | Scale-free topology confirmed via Clauset MLE |
| **Small-World Index** | $\sigma = 24.52 \gg 1$ | High clustering ($C=0.3025$) with short paths ($L=2.7667$) |
| **Link Prediction** | RA: 100% precision @ top-100 | Resource Allocation penalizes indiscriminate crawlers |
| **Complex Contagion** | Critical tipping point at 8% | Targeted PageRank seeding reaches 97.6% cascade |
| **Ergodic Random Walk** | TVD = 0.0343 at 200k steps | Empirical visit frequencies match stationary PageRank ($R^2=0.9942$) |

---

## 📜 Citation & License

This project is licensed under the [MIT License](LICENSE).

If you use this codebase, dataset, or paper in your research, please cite:

```bibtex
@inproceedings{sooriya2026peertube,
  title={Topological Invariants, Centrality Asymmetry, and Complex Contagion in Federated Social Networks: A Directed Graph Structural Analysis},
  author={Sooriya, B. and Prasanth, R. and Derin, B. C. and Aravind, S.},
  institution={Thiagarajar College of Engineering},
  year={2026}
}
```
