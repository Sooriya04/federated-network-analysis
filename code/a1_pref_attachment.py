"""
Assignment I - Task 6: Preferential Attachment and Power-Law Fitting
- Fits power-law distribution to degree using the `powerlaw` package (reports alpha, xmin).
- Compares power-law fit against lognormal alternative via log-likelihood ratio.
- Synthesizes Barabási-Albert (BA) scale-free graph with matching N and average degree m.
- Compares degree distributions, average clustering coefficient, and average path length.
- Exports metrics to CSV and saves high-resolution publication figures.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import powerlaw
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

def analyze_preferential_attachment(dataset_key: str = ACTIVE_DATASET):
    """
    Fits empirical degree distribution using powerlaw package.
    Simulates Barabási-Albert model and compares macroscopic network invariants.
    """
    print(f"\n============================================================")
    print(f"TASK 6: PREFERENTIAL ATTACHMENT & POWER-LAW ANALYSIS ({dataset_key})")
    print(f"============================================================")

    G, G_wcc, _, _ = load_and_preprocess(dataset_key)

    # Convert to undirected projection for clustering and path length comparisons
    G_undir = G.to_undirected()
    # Remove self-loops if any
    G_undir.remove_edges_from(nx.selfloop_edges(G_undir))

    # Largest connected component of undirected graph
    lcc_nodes = max(nx.connected_components(G_undir), key=len)
    G_lcc = G_undir.subgraph(lcc_nodes).copy()

    N_real = G_lcc.number_of_nodes()
    M_real = G_lcc.number_of_edges()
    degrees_real = [d for _, d in G_lcc.degree()]
    avg_k_real = 2.0 * M_real / N_real
    m_ba = max(1, int(round(avg_k_real / 2.0)))

    print(f"\nEmpirical Graph Statistics (Largest Component):")
    print(f"  Nodes (N): {N_real:,}")
    print(f"  Edges (M): {M_real:,}")
    print(f"  Average Degree <k>: {avg_k_real:.2f}")
    print(f"  Barabási-Albert attachment parameter m: {m_ba}")

    # 1. Power-Law Fit using powerlaw package
    print(f"\nFitting empirical degree distribution via powerlaw package...")
    t0 = time.time()
    fit = powerlaw.Fit(degrees_real, discrete=True, verbose=False)
    alpha = float(fit.alpha)
    xmin = float(fit.xmin)
    sigma_alpha = float(fit.sigma)

    # Compare power-law vs lognormal
    R_lognorm, p_lognorm = fit.distribution_compare("power_law", "lognormal")
    R_exp, p_exp = fit.distribution_compare("power_law", "exponential")

    t_fit = time.time() - t0
    print(f"[Done] Fit completed in {t_fit:.3f}s:")
    print(f"  Power-law exponent (alpha): {alpha:.4f} +/- {sigma_alpha:.4f}")
    print(f"  Lower cutoff threshold (xmin): {xmin:.2f}")
    print(f"  Log-likelihood ratio vs Lognormal: R={R_lognorm:.4f}, p={p_lognorm:.4f}")
    print(f"  Log-likelihood ratio vs Exponential: R={R_exp:.4f}, p={p_exp:.4f}")

    # 2. Generate Barabási-Albert Graph
    print(f"\nGenerating Barabási-Albert model G_BA(N={N_real}, m={m_ba}, seed={RANDOM_SEED})...")
    t0 = time.time()
    G_ba = nx.barabasi_albert_graph(N_real, m_ba, seed=RANDOM_SEED)
    degrees_ba = [d for _, d in G_ba.degree()]
    t_ba_gen = time.time() - t0
    print(f"[Done] BA graph generated in {t_ba_gen:.3f}s: {G_ba.number_of_nodes()} nodes, {G_ba.number_of_edges()} edges")

    # Fit BA degree distribution as reference
    fit_ba = powerlaw.Fit(degrees_ba, discrete=True, verbose=False)

    # 3. Compute Clustering and Path Length Invariants
    print("\nComputing Average Clustering Coefficient...")
    t0 = time.time()
    C_real = nx.average_clustering(G_lcc)
    C_ba = nx.average_clustering(G_ba)
    print(f"  Real C: {C_real:.4f} vs BA C: {C_ba:.4f} (took {time.time() - t0:.2f}s)")

    print("\nComputing Average Shortest Path Length...")
    t0 = time.time()
    # On 1500 nodes exact shortest path length takes ~1.5s
    L_real = nx.average_shortest_path_length(G_lcc)
    L_ba = nx.average_shortest_path_length(G_ba)
    print(f"  Real L: {L_real:.4f} vs BA L: {L_ba:.4f} (took {time.time() - t0:.2f}s)")

    # 4. Save Metrics Comparison Table
    comparison_data = {
        "Property / Metric": [
            "Nodes (N)",
            "Edges (M)",
            "Average Degree <k>",
            "Average Clustering Coefficient (C)",
            "Average Shortest Path Length (L)",
            "Power-Law Exponent (alpha)",
            "Power-Law Lower Bound (xmin)",
            "Standard Error of Alpha (sigma)",
            "Log-Likelihood Ratio vs Lognormal (R, p)",
            "Log-Likelihood Ratio vs Exponential (R, p)"
        ],
        "Real Network (Empirical)": [
            N_real,
            M_real,
            round(avg_k_real, 2),
            round(C_real, 4),
            round(L_real, 4),
            round(alpha, 4),
            round(xmin, 2),
            round(sigma_alpha, 4),
            f"R={R_lognorm:.3f}, p={p_lognorm:.3f}",
            f"R={R_exp:.3f}, p={p_exp:.3f}"
        ],
        "Barabási-Albert Model (BA)": [
            G_ba.number_of_nodes(),
            G_ba.number_of_edges(),
            round(2.0 * G_ba.number_of_edges() / G_ba.number_of_nodes(), 2),
            round(C_ba, 4),
            round(L_ba, 4),
            round(fit_ba.alpha, 4),
            round(fit_ba.xmin, 2),
            round(fit_ba.sigma, 4),
            "N/A",
            "N/A"
        ]
    }
    df_comparison = pd.DataFrame(comparison_data)
    csv_path = OUTPUTS_DIR / f"{dataset_key}_preferential_attachment_comparison.csv"
    df_comparison.to_csv(csv_path, index=False)
    print(f"\nSaved comparison metrics to: {csv_path}")

    # 5. Publication Figure: Degree Distributions & Power-Law Fit
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Plot 1: Linear Degree Histogram
    ax1 = axes[0]
    ax1.hist(degrees_real, bins=40, color="#1f77b4", alpha=0.7, edgecolor="black", label=f"Real Network (Mean={avg_k_real:.1f})")
    ax1.hist(degrees_ba, bins=40, color="#ff7f0e", alpha=0.6, edgecolor="black", label=f"Barabasi-Albert (m={m_ba})")
    ax1.set_title("Degree Distribution (Linear Scale)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Degree (k)", fontsize=11)
    ax1.set_ylabel("Node Frequency", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Log-Log Degree Complementary Cumulative Distribution (CCDF)
    ax2 = axes[1]
    fit.plot_ccdf(ax=ax2, color="#1f77b4", linewidth=2.5, label="Empirical Data (CCDF)")
    fit.power_law.plot_ccdf(ax=ax2, color="#d62728", linestyle="--", linewidth=2, label=f"Power-Law Fit ($\\alpha={alpha:.2f}$, $x_{{min}}={xmin:.1f}$)")
    fit_ba.plot_ccdf(ax=ax2, color="#ff7f0e", linestyle=":", linewidth=2, label=f"BA Model Fit ($\\alpha={fit_ba.alpha:.2f}$)")

    ax2.set_title("Degree Distribution (Log-Log CCDF) & Power-Law Fit", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Degree $k$", fontsize=11)
    ax2.set_ylabel("$P(K \\geq k)$", fontsize=11)
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = OUTPUTS_DIR / f"{dataset_key}_fig1_degree_distribution_powerlaw.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved publication figure to: {fig_path}")

    return df_comparison, alpha, xmin, C_real, C_ba, L_real, L_ba

if __name__ == "__main__":
    analyze_preferential_attachment()
