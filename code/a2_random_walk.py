"""
Assignment II - Task 3: Random Walk and Random Walk with Restart (RWR)
- Implements Random Surfer Walk (teleporting to uniform nodes with prob 1 - alpha)
  and Random Walk with Restart (RWR / Personalized Walk teleporting to seed nodes).
- Simulates walk trajectories across varying step lengths (10^3 to 2*10^5 steps).
- Compares empirical visitation frequencies against theoretical PageRank stationary distribution.
- Tracks convergence via Total Variation Distance (TVD) and Spearman rank correlation.
- Exports metrics to CSV and generates high-resolution convergence figures.
"""

import sys
import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
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

def simulate_random_walk(G: nx.DiGraph, total_steps: int = 200000, teleport_prob: float = 0.15, restart_nodes: list = None, seed: int = RANDOM_SEED) -> tuple:
    """
    Simulates a random walk on directed graph G with teleportation probability (1 - alpha).
    - If restart_nodes is None: Uniform teleportation (PageRank random surfer model).
    - If restart_nodes is provided: Teleportation back to restart_nodes (Personalized PageRank / RWR).
    Returns:
        visit_counts: dict mapping node -> number of visits
        checkpoints: dict mapping step T -> empirical frequency vector
    """
    random.seed(seed)
    nodes = list(G.nodes())
    N = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Pre-extract out-neighbors for rapid transitions
    out_neighbors = {n: list(G.successors(n)) for n in nodes}

    # Initial node
    if restart_nodes:
        current_node = random.choice(restart_nodes)
    else:
        current_node = random.choice(nodes)

    counts = np.zeros(N, dtype=int)
    checkpoint_steps = [1000, 5000, 10000, 25000, 50000, 100000, 150000, 200000]
    checkpoint_steps = [s for s in checkpoint_steps if s <= total_steps]
    if total_steps not in checkpoint_steps:
        checkpoint_steps.append(total_steps)

    checkpoints = {}

    for step in range(1, total_steps + 1):
        counts[node_to_idx[current_node]] += 1

        if step in checkpoint_steps:
            checkpoints[step] = counts / float(step)

        # Transition decision: Teleportation vs Follow Out-Edge
        if random.random() < teleport_prob:
            # Teleport
            if restart_nodes:
                current_node = random.choice(restart_nodes)
            else:
                current_node = random.choice(nodes)
        else:
            nbrs = out_neighbors[current_node]
            if nbrs:
                current_node = random.choice(nbrs)
            else:
                # Dead-end handling: teleport
                if restart_nodes:
                    current_node = random.choice(restart_nodes)
                else:
                    current_node = random.choice(nodes)

    return {nodes[i]: counts[i] for i in range(N)}, checkpoints

def run_random_walk_analysis(dataset_key: str = ACTIVE_DATASET):
    """
    Executes Random Walk and RWR, compares against theoretical stationary distributions,
    and proves convergence.
    """
    print(f"\n============================================================")
    print(f"TASK II.3: RANDOM WALK & RESTART ANALYSIS ({dataset_key})")
    print(f"============================================================")

    G, G_wcc, _, _ = load_and_preprocess(dataset_key)
    nodes = list(G.nodes())
    N = len(nodes)

    # 1. Theoretical PageRank Baselines
    print("\nComputing theoretical stationary PageRank distribution (alpha=0.85)...")
    t0 = time.time()
    pr_theoretical = nx.pagerank(G, alpha=0.85, max_iter=2000, tol=1e-08)
    pr_vec = np.array([pr_theoretical[n] for n in nodes])
    print(f"[Done] Theoretical PageRank computed in {time.time() - t0:.2f}s")

    # Top seed nodes for Personalized Walk with Restart (RWR)
    in_degs = dict(G.in_degree())
    top_seeds = sorted(in_degs, key=in_degs.get, reverse=True)[:3]
    print(f"Personalized RWR Target Seeds: {top_seeds}")

    # Compute theoretical Personalized PageRank for comparison
    ppr_personalization = {n: (1.0 / len(top_seeds) if n in top_seeds else 0.0) for n in nodes}
    ppr_theoretical = nx.pagerank(G, alpha=0.85, personalization=ppr_personalization, max_iter=2000, tol=1e-08)
    ppr_vec = np.array([ppr_theoretical[n] for n in nodes])

    # 2. Simulate Random Walk (PageRank Surfer Model, 200,000 steps)
    total_steps = 200000
    print(f"\nSimulating PageRank Surfer Random Walk ({total_steps:,} steps, teleport_prob=0.15)...")
    t0 = time.time()
    srw_counts, srw_checkpoints = simulate_random_walk(
        G, total_steps=total_steps, teleport_prob=0.15, restart_nodes=None, seed=RANDOM_SEED
    )
    print(f"[Done] Walk completed in {time.time() - t0:.2f}s")

    # 3. Simulate Random Walk with Restart (RWR, 200,000 steps)
    print(f"\nSimulating Random Walk with Restart (RWR) from seeds ({total_steps:,} steps)...")
    t0 = time.time()
    rwr_counts, rwr_checkpoints = simulate_random_walk(
        G, total_steps=total_steps, teleport_prob=0.15, restart_nodes=top_seeds, seed=RANDOM_SEED + 1
    )
    print(f"[Done] RWR completed in {time.time() - t0:.2f}s")

    # 4. Convergence Analysis (Total Variation Distance: TVD = 0.5 * sum |f_u - pi_u|)
    print("\nConvergence Evaluation over Steps:")
    convergence_records = []
    for step in sorted(srw_checkpoints.keys()):
        dist_vec = srw_checkpoints[step]
        tvd = 0.5 * np.sum(np.abs(dist_vec - pr_vec))
        rho, _ = spearmanr(dist_vec, pr_vec)
        r, _ = pearsonr(dist_vec, pr_vec)

        # Also RWR vs Personalized PageRank
        rwr_dist = rwr_checkpoints[step]
        tvd_rwr = 0.5 * np.sum(np.abs(rwr_dist - ppr_vec))
        r_rwr, _ = pearsonr(rwr_dist, ppr_vec)

        convergence_records.append({
            "Steps": step,
            "TVD_PageRank": round(float(tvd), 4),
            "Spearman_rho_PR": round(float(rho), 4),
            "Pearson_r_PR": round(float(r), 4),
            "TVD_RWR": round(float(tvd_rwr), 4),
            "Pearson_r_RWR": round(float(r_rwr), 4)
        })
        print(f"  Step {step:6d} | TVD = {tvd:.4f} | Spearman rho = {rho:.4f} | Pearson r = {r:.4f} (RWR Pearson r = {r_rwr:.4f})")

    df_conv = pd.DataFrame(convergence_records)
    conv_csv = OUTPUTS_DIR / f"{dataset_key}_random_walk_convergence.csv"
    df_conv.to_csv(conv_csv, index=False)
    print(f"\nSaved convergence data to: {conv_csv}")

    # Final Visitation Summary Table
    df_walk_summary = pd.DataFrame({
        "node": nodes,
        "theoretical_pagerank": [pr_theoretical[n] for n in nodes],
        "empirical_srw_freq": [srw_counts[n] / total_steps for n in nodes],
        "theoretical_ppr": [ppr_theoretical[n] for n in nodes],
        "empirical_rwr_freq": [rwr_counts[n] / total_steps for n in nodes]
    }).sort_values(by="theoretical_pagerank", ascending=False).reset_index(drop=True)

    summary_csv = OUTPUTS_DIR / f"{dataset_key}_random_walk_visits_top.csv"
    df_walk_summary.head(20).to_csv(summary_csv, index=False)
    print(f"Saved top visit comparison table to: {summary_csv}")

    # 5. Publication Visualization: Convergence and Correlation Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Plot 1: Total Variation Distance Convergence Curve
    ax1 = axes[0]
    ax1.plot(df_conv["Steps"], df_conv["TVD_PageRank"], marker="o", color="#d62728", linewidth=2.5, label="PageRank Walk (TVD to $\\pi$)")
    ax1.plot(df_conv["Steps"], df_conv["TVD_RWR"], marker="s", linestyle="--", color="#1f77b4", linewidth=2.0, label="RWR (TVD to Personalized $\\pi$)")
    ax1.set_title("Random Walk Convergence to Stationary Distribution", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Walk Steps ($T$)", fontsize=11)
    ax1.set_ylabel("Total Variation Distance $TVD(f_T, \\pi)$", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Scatter Empirical Visit Frequencies vs Theoretical Stationary Distribution
    ax2 = axes[1]
    final_r = df_conv.iloc[-1]["Pearson_r_PR"]
    final_rho = df_conv.iloc[-1]["Spearman_rho_PR"]
    ax2.scatter(
        df_walk_summary["theoretical_pagerank"],
        df_walk_summary["empirical_srw_freq"],
        alpha=0.6,
        color="#1f77b4",
        edgecolor="black",
        s=30,
        label=f"Empirical Frequencies ($r={final_r:.3f}, \\rho={final_rho:.3f}$)"
    )
    max_val = max(max(pr_vec), max(df_walk_summary["empirical_srw_freq"]))
    ax2.plot([0, max_val], [0, max_val], linestyle="--", color="red", linewidth=2, label="Identity Line ($y = x$)")
    ax2.set_title("Empirical Random Walk Frequency vs Theoretical PageRank", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Theoretical PageRank Stationary Probability $\\pi_u$", fontsize=11)
    ax2.set_ylabel("Empirical Random Walk Frequency $f_u$", fontsize=11)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Random Walk Stationary Convergence (Ergodic Theorem Validation)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = OUTPUTS_DIR / f"{dataset_key}_fig7_random_walk_convergence.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved random walk publication figure to: {fig_path}")

    return df_conv, df_walk_summary

if __name__ == "__main__":
    run_random_walk_analysis()
