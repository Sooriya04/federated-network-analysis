"""
Assignment I - Task 1: Preprocessing and Network Construction
- Loads raw directed dataset (Fedivertex PeerTube / SNAP wiki-Vote)
- Handles missing values, removes self-loops, and eliminates duplicates
- Constructs directed graph (nx.DiGraph)
- Analyzes weakly connected components (WCC) and strongly connected components (SCC)
- Exports cleaned edge lists and summary tables
- Documents component usage rationale for downstream tasks
"""

import sys
import time
from pathlib import Path
import pandas as pd
import networkx as nx

# Add project root to sys.path to enable src imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ACTIVE_DATASET,
    DATASET_CONFIGS,
    CLEANED_DATA_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED
)

def load_and_preprocess(dataset_key: str = ACTIVE_DATASET):
    """
    Loads raw dataset, performs rigorous cleaning, and computes connectivity statistics.
    Returns:
        G_cleaned: nx.DiGraph of the full cleaned network
        G_wcc: nx.DiGraph of the largest weakly connected component
        G_scc: nx.DiGraph of the largest strongly connected component
        summary_df: pd.DataFrame with before/after statistics
    """
    t0 = time.time()
    cfg = DATASET_CONFIGS[dataset_key]
    raw_path = cfg["raw_path"]
    print(f"============================================================")
    print(f"PREPROCESSING: Loading '{cfg['name']}' from:")
    print(f"  {raw_path}")
    print(f"============================================================")

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")

    # 1. Load Raw Edge Data
    if cfg["has_header"]:
        df_raw = pd.read_csv(raw_path, delimiter=cfg["delimiter"])
        src_col = cfg["source_col"]
        tgt_col = cfg["target_col"]
    else:
        df_raw = pd.read_csv(
            raw_path,
            delimiter=cfg["delimiter"],
            comment="#",
            header=None,
            names=["Source", "Target"]
        )
        src_col = "Source"
        tgt_col = "Target"

    raw_edge_count = len(df_raw)
    raw_unique_nodes = len(set(df_raw[src_col].dropna()).union(set(df_raw[tgt_col].dropna())))

    print(f"Raw graph loaded in {time.time() - t0:.2f}s:")
    print(f"  Raw Edge Rows: {raw_edge_count:,}")
    print(f"  Raw Unique Nodes: {raw_unique_nodes:,}")

    # 2. Handle Missing Values
    null_rows = df_raw[[src_col, tgt_col]].isnull().any(axis=1).sum()
    df_clean = df_raw.dropna(subset=[src_col, tgt_col]).copy()
    # Ensure node identifiers are strings for consistency
    df_clean[src_col] = df_clean[src_col].astype(str).str.strip()
    df_clean[tgt_col] = df_clean[tgt_col].astype(str).str.strip()
    # Filter out empty strings if any
    df_clean = df_clean[(df_clean[src_col] != "") & (df_clean[tgt_col] != "")]

    # 3. Detect and Remove Self-Loops
    self_loops_mask = df_clean[src_col] == df_clean[tgt_col]
    self_loop_count = int(self_loops_mask.sum())
    df_clean = df_clean[~self_loops_mask]

    # 4. Detect and Remove Duplicate Directed Edges
    duplicate_count = int(df_clean.duplicated(subset=[src_col, tgt_col]).sum())
    df_clean = df_clean.drop_duplicates(subset=[src_col, tgt_col]).reset_index(drop=True)

    cleaned_edge_count = len(df_clean)
    cleaned_nodes = sorted(list(set(df_clean[src_col]).union(set(df_clean[tgt_col]))))
    cleaned_node_count = len(cleaned_nodes)

    print(f"\nData Cleaning Summary:")
    print(f"  Null / Missing rows removed: {null_rows}")
    print(f"  Self-loops removed (u == v): {self_loop_count}")
    print(f"  Duplicate directed edges removed: {duplicate_count}")
    print(f"  Cleaned Directed Graph: {cleaned_node_count:,} nodes, {cleaned_edge_count:,} edges")

    # 5. Build Directed Graph in NetworkX
    G = nx.DiGraph()
    # Optional edge attributes (e.g. Weight)
    if "Weight" in df_clean.columns:
        edges = list(zip(df_clean[src_col], df_clean[tgt_col], [{"weight": float(w)} for w in df_clean["Weight"]]))
        G.add_edges_from(edges)
    else:
        edges = list(zip(df_clean[src_col], df_clean[tgt_col]))
        G.add_edges_from(edges)

    # Attach node metadata if available
    if cfg["metadata_path"] and cfg["metadata_path"].exists():
        try:
            df_meta = pd.read_csv(cfg["metadata_path"])
            id_col = "host" if "host" in df_meta.columns else ("Id" if "Id" in df_meta.columns else df_meta.columns[0])
            df_meta[id_col] = df_meta[id_col].astype(str).str.strip()
            meta_dict = df_meta.set_index(id_col).to_dict(orient="index")
            nx.set_node_attributes(G, meta_dict)
            print(f"  Attached metadata attributes for {len(meta_dict):,} nodes from {cfg['metadata_path'].name}")
        except Exception as e:
            print(f"  [Notice] Metadata loading skipped: {e}")

    # 6. Connected Components Analysis
    wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    sccs = sorted(nx.strongly_connected_components(G), key=len, reverse=True)

    largest_wcc_nodes = wccs[0]
    G_wcc = G.subgraph(largest_wcc_nodes).copy()

    largest_scc_nodes = sccs[0]
    G_scc = G.subgraph(largest_scc_nodes).copy()

    wcc_coverage_nodes = len(largest_wcc_nodes) / cleaned_node_count * 100
    wcc_coverage_edges = G_wcc.number_of_edges() / cleaned_edge_count * 100
    scc_coverage_nodes = len(largest_scc_nodes) / cleaned_node_count * 100
    scc_coverage_edges = G_scc.number_of_edges() / cleaned_edge_count * 100

    print(f"\nComponent Decomposition:")
    print(f"  Weakly Connected Components (WCC): {len(wccs)}")
    print(f"    Largest WCC: {G_wcc.number_of_nodes():,} nodes ({wcc_coverage_nodes:.2f}%), {G_wcc.number_of_edges():,} edges ({wcc_coverage_edges:.2f}%)")
    print(f"  Strongly Connected Components (SCC): {len(sccs)}")
    print(f"    Largest SCC: {G_scc.number_of_nodes():,} nodes ({scc_coverage_nodes:.2f}%), {G_scc.number_of_edges():,} edges ({scc_coverage_edges:.2f}%)")

    # 7. Save Cleaned Edgelist
    cleaned_edgelist_path = CLEANED_DATA_DIR / f"{dataset_key}_cleaned_edges.csv"
    df_clean[[src_col, tgt_col]].rename(columns={src_col: "Source", tgt_col: "Target"}).to_csv(
        cleaned_edgelist_path, index=False
    )
    print(f"\nSaved cleaned edge list to: {cleaned_edgelist_path}")

    # 8. Save Preprocessing Summary Table
    summary_data = {
        "Metric": [
            "Dataset Identifier",
            "Dataset Name",
            "Dataset Citation",
            "Graph Type",
            "Raw Total Edge Records",
            "Raw Unique Node Identifiers",
            "Missing / Null Records Removed",
            "Self-Loops Removed",
            "Duplicate Directed Edges Removed",
            "Cleaned Total Nodes",
            "Cleaned Total Directed Edges",
            "Cleaned Average Degree (In = Out)",
            "Cleaned Graph Density",
            "Total Weakly Connected Components (WCC)",
            "Largest WCC Nodes",
            "Largest WCC Edges",
            "Largest WCC Node Fraction (%)",
            "Total Strongly Connected Components (SCC)",
            "Largest SCC Nodes",
            "Largest SCC Edges",
            "Largest SCC Node Fraction (%)",
            "Degree Centrality Target Graph",
            "PageRank & HITS Target Graph",
            "Closeness & Betweenness Target Graph",
            "Small-World & Path Analysis Target Graph",
            "Link Prediction Target Graph",
            "Granovetter Cascades Target Graph"
        ],
        "Value": [
            dataset_key,
            cfg["name"],
            cfg["citation"],
            "Directed Graph (nx.DiGraph)",
            raw_edge_count,
            raw_unique_nodes,
            null_rows,
            self_loop_count,
            duplicate_count,
            cleaned_node_count,
            cleaned_edge_count,
            round(cleaned_edge_count / cleaned_node_count, 4),
            round(cleaned_edge_count / (cleaned_node_count * (cleaned_node_count - 1)), 6),
            len(wccs),
            G_wcc.number_of_nodes(),
            G_wcc.number_of_edges(),
            round(wcc_coverage_nodes, 2),
            len(sccs),
            G_scc.number_of_nodes(),
            G_scc.number_of_edges(),
            round(scc_coverage_nodes, 2),
            "Cleaned Full Graph G (captures global in/out nominations without truncation)",
            "Cleaned Full Graph G (stochastic transition matrix operates on full link structure)",
            "Largest WCC (prevents infinite distances and division by zero across disconnected components)",
            "Largest WCC (standard complex-network benchmark to compute finite characteristic path lengths)",
            "Largest WCC (ensures negative non-edge sampling occurs over reachable topological domain)",
            "Largest WCC (avoids artificial cascade halts at isolated singleton boundaries)"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = OUTPUTS_DIR / f"{dataset_key}_preprocessing_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved preprocessing summary to: {summary_csv_path}")

    return G, G_wcc, G_scc, summary_df

if __name__ == "__main__":
    load_and_preprocess()
