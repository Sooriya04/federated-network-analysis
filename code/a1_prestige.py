"""
Assignment I - Task 3: Prestige Measures in Directed Networks
- Computes In-Degree Prestige, Proximity Prestige, and Rank Prestige (Katz-based).
- Formulates each prestige measure according to Wasserman & Faust (1994).
- Provides one-line theoretical explanations for each.
- Exports Top-10 tables and saves results to CSV.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
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

PRESTIGE_DEFINITIONS = {
    "in_degree_prestige": "Measures direct actor popularity or endorsement as the proportion of all other actors in the network that directly nominate or follow actor i.",
    "proximity_prestige": "Measures how closely reachable an actor is from all members of its influence domain, scaled by the proportion of the network that can reach it.",
    "rank_prestige": "Measures actor prominence recursively based on the status of endorsements received, where nominations from already prestigious actors contribute more weight."
}

def compute_proximity_prestige(G: nx.DiGraph) -> dict:
    """
    Computes Wasserman-Faust Proximity Prestige for all nodes in a directed graph.
    P_P(i) = [ |I(i)| / (N - 1) ] / [ sum_{j in I(i)} d(j, i) / |I(i)| ]
           = |I(i)|^2 / [ (N - 1) * sum_{j in I(i)} d(j, i) ]
    where I(i) is the influence domain (all actors with a directed path to i).
    """
    N = G.number_of_nodes()
    if N <= 1:
        return {n: 0.0 for n in G.nodes()}

    # Reverse graph: shortest paths from i in G_rev correspond to paths to i in G
    G_rev = G.reverse()
    proximity_scores = {}

    for node in G.nodes():
        # Compute lengths of all shortest paths targeting node
        lengths = nx.single_source_shortest_path_length(G_rev, node)
        # Exclude self distance (d(node, node) = 0)
        lengths.pop(node, None)

        influence_domain_size = len(lengths)
        if influence_domain_size == 0:
            proximity_scores[node] = 0.0
        else:
            total_distance = sum(lengths.values())
            if total_distance > 0:
                proximity_scores[node] = (influence_domain_size ** 2) / ((N - 1) * total_distance)
            else:
                proximity_scores[node] = 0.0

    return proximity_scores

def compute_rank_prestige_katz(G: nx.DiGraph, alpha: float = None, beta: float = 1.0) -> dict:
    """
    Computes Katz Rank Prestige:
    p = alpha * A^T * p + beta * 1
    (I - alpha * A^T) * p = beta * 1
    where A is the adjacency matrix (A_ij = 1 if edge i -> j).
    """
    nodes = list(G.nodes())
    N = len(nodes)
    if N == 0:
        return {}

    # Adjacency matrix A (sparse CSR)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")

    # Find maximum eigenvalue of A^T (spectral radius)
    try:
        eigenvalues = spla.eigs(A.T, k=1, which="LM", return_eigenvectors=False)
        spectral_radius = float(np.abs(eigenvalues[0]))
    except Exception:
        # Fallback approximation: max row sum
        spectral_radius = float(max(np.diff(A.indptr)))

    # Set alpha strictly smaller than 1 / spectral_radius
    if alpha is None or alpha >= (1.0 / spectral_radius):
        alpha = 0.85 / max(spectral_radius, 1.0)

    # Solve linear system: (I - alpha * A^T) * p = beta * 1
    I = sp.eye(N, format="csr")
    M = I - alpha * A.T
    b = np.full(N, beta)

    try:
        p, info = spla.gmres(M, b, tol=1e-8, maxiter=1000)
        if info != 0:
            p = spla.spsolve(M.tocsc(), b)
    except Exception:
        p = spla.spsolve(M.tocsc(), b)

    # Normalize rank prestige vector to unit norm or sum to 1
    p_norm = p / np.linalg.norm(p)
    return {nodes[i]: float(p_norm[i]) for i in range(N)}

def compute_prestige(dataset_key: str = ACTIVE_DATASET):
    """
    Executes in-degree prestige, proximity prestige, and rank prestige.
    Outputs Top-10 tables and saves to CSV.
    """
    print(f"\n============================================================")
    print(f"TASK 3: PRESTIGE MEASURES ({dataset_key})")
    print(f"============================================================")

    G, G_wcc, _, _ = load_and_preprocess(dataset_key)
    N = G.number_of_nodes()

    # 1. In-Degree Prestige (Nomination Prestige)
    t0 = time.time()
    in_deg = dict(G.in_degree())
    norm = N - 1 if N > 1 else 1
    p_degree = {n: in_deg[n] / norm for n in G.nodes()}
    t_degree = time.time() - t0
    print(f"[Done] In-Degree Prestige computed in {t_degree:.3f}s")

    # 2. Proximity Prestige
    t0 = time.time()
    p_proximity = compute_proximity_prestige(G)
    t_proximity = time.time() - t0
    print(f"[Done] Proximity Prestige computed in {t_proximity:.3f}s")

    # 3. Rank Prestige (Katz Prestige)
    t0 = time.time()
    p_rank = compute_rank_prestige_katz(G)
    t_rank = time.time() - t0
    print(f"[Done] Rank Prestige (Katz) computed in {t_rank:.3f}s")

    # 4. Assemble Results DataFrame
    df_prestige = pd.DataFrame({
        "node": list(G.nodes()),
        "in_degree_prestige": [p_degree[n] for n in G.nodes()],
        "proximity_prestige": [p_proximity[n] for n in G.nodes()],
        "rank_prestige": [p_rank[n] for n in G.nodes()]
    })

    full_csv = OUTPUTS_DIR / f"{dataset_key}_prestige_all.csv"
    df_prestige.to_csv(full_csv, index=False)
    print(f"Saved complete prestige scores to: {full_csv}")

    # 5. Extract Top-10 Tables
    measures = [
        ("in_degree_prestige", "In-Degree Prestige", PRESTIGE_DEFINITIONS["in_degree_prestige"]),
        ("proximity_prestige", "Proximity Prestige", PRESTIGE_DEFINITIONS["proximity_prestige"]),
        ("rank_prestige", "Rank Prestige (Katz)", PRESTIGE_DEFINITIONS["rank_prestige"])
    ]

    top10_records = []
    for col, title, definition in measures:
        top10 = df_prestige.sort_values(by=col, ascending=False).head(10).reset_index(drop=True)
        top10_csv = OUTPUTS_DIR / f"{dataset_key}_top10_{col}.csv"
        top10[["node", col]].rename(columns={"node": "Node ID", col: "Prestige Score"}).to_csv(top10_csv, index=False)

        print(f"\n------------------------------------------------------------")
        print(f"Top-10 by {title}:")
        print(f"Definition: {definition}")
        print(f"------------------------------------------------------------")
        for rank, row in top10.iterrows():
            print(f"  Rank {rank+1:2d}: {row['node']:<35s} | {row[col]:.6f}")
            top10_records.append({
                "Prestige Measure": title,
                "Definition": definition,
                "Rank": rank + 1,
                "Node ID": row["node"],
                "Score": row[col]
            })

    df_top10_prestige = pd.DataFrame(top10_records)
    df_top10_prestige.to_csv(OUTPUTS_DIR / f"{dataset_key}_top10_prestige_combined.csv", index=False)
    print(f"\nSaved combined prestige top-10 to: {OUTPUTS_DIR / f'{dataset_key}_top10_prestige_combined.csv'}")

    return df_prestige, df_top10_prestige

if __name__ == "__main__":
    compute_prestige()
