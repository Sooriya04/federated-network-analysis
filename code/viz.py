"""
Visualization Module (Assignment I Task 8 & Assignment II Task 4)
- High-resolution (300 dpi) publication plots:
  * Readable subnetwork graph (top-k core by PageRank, sized by PR, colored by Louvain community)
  * Centrality correlation heatmap (Pearson and Spearman)
  * Degree distributions (linear and log-log)
- Gephi GEXF export with all computed node and edge properties.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ACTIVE_DATASET,
    OUTPUTS_DIR,
    RANDOM_SEED
)
from src.a1_preprocess import load_and_preprocess

def plot_centrality_correlation_heatmap(dataset_key: str = ACTIVE_DATASET):
    """
    Plots side-by-side heatmaps of Pearson and Spearman centrality correlations.
    """
    pearson_path = OUTPUTS_DIR / f"{dataset_key}_centrality_pearson_corr.csv"
    spearman_path = OUTPUTS_DIR / f"{dataset_key}_centrality_spearman_corr.csv"

    if not pearson_path.exists() or not spearman_path.exists():
        print("[Notice] Centrality correlation CSVs not found. Run a1_centrality.py first.")
        return

    df_pearson = pd.read_csv(pearson_path, index_col=0)
    df_spearman = pd.read_csv(spearman_path, index_col=0)

    labels = ["In-Degree", "Out-Degree", "Closeness", "Betweenness", "Eigenvector"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # 1. Pearson Heatmap
    im1 = axes[0].imshow(df_pearson.values, cmap="YlGnBu", vmin=0, vmax=1)
    axes[0].set_title("Centrality Pearson Correlation Matrix", fontsize=12, fontweight="bold")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    axes[0].set_yticklabels(labels, fontsize=10)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = df_pearson.values[i, j]
            color = "white" if val > 0.6 else "black"
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    # 2. Spearman Heatmap
    im2 = axes[1].imshow(df_spearman.values, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Centrality Spearman Rank Correlation Matrix", fontsize=12, fontweight="bold")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    axes[1].set_yticklabels(labels, fontsize=10)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = df_spearman.values[i, j]
            color = "white" if val > 0.6 else "black"
            axes[1].text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    plt.tight_layout()
    fig_path = OUTPUTS_DIR / f"{dataset_key}_fig3_centrality_correlation_heatmaps.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved centrality heatmap to: {fig_path}")

def plot_readable_network_core(dataset_key: str = ACTIVE_DATASET, top_k: int = 70):
    """
    Plots a readable, publication-grade network diagram of the top-k core nodes by PageRank.
    - Node size scaled by PageRank.
    - Node color mapped by Louvain community partition.
    - Prominent nodes labeled.
    """
    G, _, _, _ = load_and_preprocess(dataset_key)

    pr_path = OUTPUTS_DIR / f"{dataset_key}_pagerank_all.csv"
    if pr_path.exists():
        df_pr = pd.read_csv(pr_path)
        top_nodes = df_pr.sort_values(by="pagerank_085", ascending=False)["node"].head(top_k).tolist()
        pr_dict = df_pr.set_index("node")["pagerank_085"].to_dict()
    else:
        pr_dict = nx.pagerank(G, alpha=0.85)
        top_nodes = sorted(pr_dict, key=pr_dict.get, reverse=True)[:top_k]

    # Induce subgraph on top-k nodes
    H = G.subgraph(top_nodes).copy()
    print(f"Core Subgraph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    # Detect communities via Louvain
    H_undir = H.to_undirected()
    communities = nx.community.louvain_communities(H_undir, seed=RANDOM_SEED)
    node_to_comm = {}
    for comm_id, c in enumerate(communities):
        for node in c:
            node_to_comm[node] = comm_id

    # Layout
    pos = nx.spring_layout(H, k=0.45, iterations=120, seed=RANDOM_SEED)

    # Node sizes and colors
    max_pr = max([pr_dict.get(n, 0.001) for n in H.nodes()])
    node_sizes = [max(80, int((pr_dict.get(n, 0.001) / max_pr) * 800)) for n in H.nodes()]
    node_colors = [node_to_comm.get(n, 0) for n in H.nodes()]

    plt.figure(figsize=(12, 10), dpi=300)
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(1, len(communities)))

    # Draw edges with alpha
    nx.draw_networkx_edges(H, pos, alpha=0.15, edge_color="#444444", arrows=True, arrowsize=8, width=0.8)

    # Draw nodes
    nx.draw_networkx_nodes(
        H, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=cmap,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.8
    )

    # Label only top 15 highest PageRank nodes for clean readability
    top15_labels = {n: n.replace("[DOT]", ".").replace("www.", "") for n in top_nodes[:15]}
    nx.draw_networkx_labels(
        H, pos,
        labels=top15_labels,
        font_size=8,
        font_weight="bold",
        font_color="#111111",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")
    )

    plt.title(
        f"Core Network Architecture (Top-{top_k} Nodes by PageRank)\n"
        f"Node Size $\\propto$ PageRank ($d=0.85$), Color = Louvain Community ({len(communities)} Clusters)",
        fontsize=13, fontweight="bold", pad=15
    )
    plt.axis("off")
    plt.tight_layout()

    fig_path = OUTPUTS_DIR / f"{dataset_key}_fig4_network_core_pagerank_communities.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved core network visualization to: {fig_path}")

def export_gephi_gexf(dataset_key: str = ACTIVE_DATASET):
    """
    Exports the complete directed graph to Gephi GEXF format, attaching all
    computed centralities, prestige, PageRank, HITS, and community partitions as node attributes.
    """
    print(f"\nExporting Gephi GEXF file with computed attributes...")
    G, _, _, _ = load_and_preprocess(dataset_key)

    # Load computed attributes from CSVs if available
    cent_path = OUTPUTS_DIR / f"{dataset_key}_centrality_all.csv"
    pr_path = OUTPUTS_DIR / f"{dataset_key}_pagerank_all.csv"
    hits_path = OUTPUTS_DIR / f"{dataset_key}_hits_all.csv"
    prestige_path = OUTPUTS_DIR / f"{dataset_key}_prestige_all.csv"

    # Compute Louvain communities
    G_undir = G.to_undirected()
    communities = nx.community.louvain_communities(G_undir, seed=RANDOM_SEED)
    comm_dict = {}
    for c_id, c in enumerate(communities):
        for n in c:
            comm_dict[n] = int(c_id)

    # Node attributes dictionary
    node_attrs = {n: {"community": comm_dict.get(n, 0)} for n in G.nodes()}

    if cent_path.exists():
        df_c = pd.read_csv(cent_path).set_index("node")
        for n in G.nodes():
            if n in df_c.index:
                node_attrs[n]["in_degree"] = int(df_c.loc[n, "in_degree"])
                node_attrs[n]["out_degree"] = int(df_c.loc[n, "out_degree"])
                node_attrs[n]["closeness"] = float(df_c.loc[n, "closeness"])
                node_attrs[n]["betweenness"] = float(df_c.loc[n, "betweenness"])
                node_attrs[n]["eigenvector"] = float(df_c.loc[n, "eigenvector"])

    if pr_path.exists():
        df_p = pd.read_csv(pr_path).set_index("node")
        for n in G.nodes():
            if n in df_p.index:
                node_attrs[n]["pagerank_085"] = float(df_p.loc[n, "pagerank_085"])
                node_attrs[n]["pagerank_050"] = float(df_p.loc[n, "pagerank_050"])
                node_attrs[n]["pagerank_099"] = float(df_p.loc[n, "pagerank_099"])

    if hits_path.exists():
        df_h = pd.read_csv(hits_path).set_index("node")
        for n in G.nodes():
            if n in df_h.index:
                node_attrs[n]["hub_score"] = float(df_h.loc[n, "hub_score"])
                node_attrs[n]["authority_score"] = float(df_h.loc[n, "authority_score"])

    if prestige_path.exists():
        df_prest = pd.read_csv(prestige_path).set_index("node")
        for n in G.nodes():
            if n in df_prest.index:
                node_attrs[n]["proximity_prestige"] = float(df_prest.loc[n, "proximity_prestige"])
                node_attrs[n]["rank_prestige"] = float(df_prest.loc[n, "rank_prestige"])

    # Set attributes
    nx.set_node_attributes(G, node_attrs)

    # Export GEXF
    gexf_path = OUTPUTS_DIR / f"{dataset_key}_network.gexf"
    nx.write_gexf(G, gexf_path)
    print(f"[Done] Exported Gephi GEXF ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) to: {gexf_path}")

def generate_all_visualizations(dataset_key: str = ACTIVE_DATASET):
    """Generates all visualizations and Gephi exports."""
    plot_centrality_correlation_heatmap(dataset_key)
    plot_readable_network_core(dataset_key)
    export_gephi_gexf(dataset_key)

if __name__ == "__main__":
    generate_all_visualizations()
