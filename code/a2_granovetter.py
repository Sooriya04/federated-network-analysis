"""
Assignment II - Task 2: Granovetter Threshold Model Implementation
- Implements the Granovetter (1978) threshold diffusion model from scratch.
- Run (a): Classic synthetic collective action example demonstrating complete cascade vs abrupt failure.
- Run (b): Real network simulation with Gaussian distributed thresholds N(mu=0.30, sigma=0.10).
- Compares Random Seeds vs Top-PageRank Influential Seeds across seed fractions.
- Analyzes temporal cascade dynamics (adoption over time) and phase transitions (tipping point).
- Exports metrics to CSV and generates publication-grade trajectory and tipping-point figures.
"""

import sys
import time
import random
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

class GranovetterThresholdModel:
    """
    Granovetter (1978) Complex Contagion Threshold Diffusion Model.
    Each node i has threshold theta_i in [0, 1].
    Node i adopts if (active_neighbors / total_neighbors) >= theta_i.
    """
    def __init__(self, G: nx.DiGraph, thresholds: dict = None, default_theta: float = 0.3):
        self.G = G
        self.nodes = list(G.nodes())
        self.N = len(self.nodes)

        # Precompute in-neighbors (incoming influence in directed graphs)
        # For each node i, in_neighbors are nodes j pointing to i (j -> i)
        self.in_neighbors = {n: set(G.predecessors(n)) for n in self.nodes}
        self.in_degrees = {n: len(self.in_neighbors[n]) for n in self.nodes}

        if thresholds is not None:
            self.thresholds = thresholds
        else:
            self.thresholds = {n: default_theta for n in self.nodes}

    def simulate(self, initial_seeds: set, max_steps: int = 100) -> tuple:
        """
        Executes cascade simulation starting from initial_seeds.
        Returns:
            trajectory: list of active node counts at each discrete time step [t=0, 1, ...]
            active_nodes: final set of active nodes
            steps_taken: number of steps until convergence
        """
        active = set(initial_seeds)
        trajectory = [len(active)]

        for step in range(max_steps):
            newly_active = set()

            for node in self.nodes:
                if node in active:
                    continue

                d_in = self.in_degrees[node]
                if d_in == 0:
                    # Isolated or source node adopts only if threshold is 0
                    if self.thresholds[node] == 0:
                        newly_active.add(node)
                    continue

                active_neighbors_count = len(self.in_neighbors[node].intersection(active))
                fraction_active = active_neighbors_count / d_in

                if fraction_active >= self.thresholds[node]:
                    newly_active.add(node)

            if not newly_active:
                # Equilibrium reached
                break

            active.update(newly_active)
            trajectory.append(len(active))

        return trajectory, active, len(trajectory) - 1

def run_classic_toy_example():
    """
    Simulates Granovetter's classic 100-person collective action example:
    - Case 1: Complete cascade (thresholds 0 to 99).
    - Case 2: Blocked cascade (threshold 1 removed and replaced by 2).
    """
    print("\n--- Granovetter Classic Toy Example (N=100) ---")
    N = 100

    # In Granovetter's original model, individuals interact in a complete mixing pool
    G_complete = nx.complete_graph(N).to_directed()
    nodes = list(G_complete.nodes())

    # Case 1: Thresholds exactly 0/100, 1/100, 2/100, ..., 99/100
    thresholds_case1 = {i: i / 100.0 for i in nodes}
    model1 = GranovetterThresholdModel(G_complete, thresholds_case1)
    seed1 = {0} # Person 0 is the initial instigator (threshold 0)
    traj1, active1, steps1 = model1.simulate(seed1)

    # Case 2: Replace person 1 (threshold 0.01) with threshold 0.02
    thresholds_case2 = dict(thresholds_case1)
    thresholds_case2[1] = 0.02 # Missing single bridge individual
    model2 = GranovetterThresholdModel(G_complete, thresholds_case2)
    traj2, active2, steps2 = model2.simulate(seed1)

    print(f"Case 1 (Full Sequence 0..99): Final Adoption = {len(active1)}/100 ({len(active1)}%) in {steps1} steps")
    print(f"Case 2 (Threshold 1 Missing): Final Adoption = {len(active2)}/100 ({len(active2)}%) in {steps2} steps")

    toy_results = pd.DataFrame({
        "Scenario": ["Full Linear Sequence (0..99)", "Single Gap at Index 1 (No 1% Threshold)"],
        "Initial_Seeds": [1, 1],
        "Final_Adoption_Count": [len(active1), len(active2)],
        "Final_Adoption_Fraction": [len(active1) / N, len(active2) / N],
        "Convergence_Steps": [steps1, steps2],
        "Cascade_Outcome": ["Global Complete Cascade (100%)", "Abrupt Cascade Arrest (1% only)"]
    })
    toy_results.to_csv(OUTPUTS_DIR / "granovetter_toy_example_results.csv", index=False)
    return traj1, traj2

def simulate_real_network_cascades(dataset_key: str = ACTIVE_DATASET):
    """
    Simulates threshold cascade on real network with Gaussian thresholds.
    Compares Random Seeds vs Top-PageRank Seeds across seed fractions.
    """
    print(f"\n============================================================")
    print(f"TASK II.2: GRANOVETTER THRESHOLD MODEL ({dataset_key})")
    print(f"============================================================")

    # Run classic toy model first
    traj_toy1, traj_toy2 = run_classic_toy_example()

    # Load real graph
    G, G_wcc, _, _ = load_and_preprocess(dataset_key)
    nodes = list(G_wcc.nodes())
    N = len(nodes)

    # Pre-fetch PageRank rankings
    pr_path = OUTPUTS_DIR / f"{dataset_key}_pagerank_all.csv"
    if pr_path.exists():
        df_pr = pd.read_csv(pr_path)
        top_pr_nodes = df_pr.sort_values(by="pagerank_085", ascending=False)["node"].tolist()
        top_pr_nodes = [n for n in top_pr_nodes if n in G_wcc]
    else:
        pr = nx.pagerank(G_wcc, alpha=0.85)
        top_pr_nodes = sorted(pr, key=pr.get, reverse=True)

    # Draw Gaussian Thresholds: N(mu=0.30, sigma=0.10) clipped to [0.05, 0.95]
    np.random.seed(RANDOM_SEED)
    raw_thresholds = np.random.normal(loc=0.30, scale=0.10, size=N)
    clipped_thresholds = np.clip(raw_thresholds, 0.05, 0.95)
    node_thresholds = {nodes[i]: float(clipped_thresholds[i]) for i in range(N)}

    model = GranovetterThresholdModel(G_wcc, node_thresholds)

    seed_fractions = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    n_reps_random = 5

    sweep_results = []
    trajectories_pr = {}
    trajectories_rnd = {}

    print(f"\nSimulating Threshold Cascades across {len(seed_fractions)} seed fractions (N={N:,})...")

    for f_seed in seed_fractions:
        k_seed = max(1, int(round(f_seed * N)))

        # 1. Influential Seeds (Top PageRank)
        pr_seeds = set(top_pr_nodes[:k_seed])
        traj_pr, active_pr, steps_pr = model.simulate(pr_seeds)
        final_pr_frac = len(active_pr) / N
        trajectories_pr[f_seed] = traj_pr

        # 2. Random Seeds (averaged over repetitions)
        random_fracs = []
        best_random_traj = None
        for rep in range(n_reps_random):
            rnd_seed = set(random.sample(nodes, k_seed))
            traj_rnd, active_rnd, _ = model.simulate(rnd_seed)
            random_fracs.append(len(active_rnd) / N)
            if rep == 0:
                best_random_traj = traj_rnd

        trajectories_rnd[f_seed] = best_random_traj
        mean_rnd_frac = float(np.mean(random_fracs))
        std_rnd_frac = float(np.std(random_fracs))

        print(f"  Seed Fraction {f_seed*100:4.1f}% ({k_seed:3d} seeds): Top-PR Adoption = {final_pr_frac*100:5.1f}% | Random Adoption = {mean_rnd_frac*100:5.1f}% (+/- {std_rnd_frac*100:4.1f}%)")

        sweep_results.append({
            "Seed_Fraction": f_seed,
            "Seed_Count": k_seed,
            "Top_PageRank_Final_Adoption": round(final_pr_frac, 4),
            "Top_PageRank_Steps": steps_pr,
            "Random_Final_Adoption_Mean": round(mean_rnd_frac, 4),
            "Random_Final_Adoption_Std": round(std_rnd_frac, 4),
            "Influential_Advantage_Ratio": round(final_pr_frac / max(mean_rnd_frac, 0.001), 2)
        })

    df_sweep = pd.DataFrame(sweep_results)
    sweep_csv = OUTPUTS_DIR / f"{dataset_key}_granovetter_cascade_sweep.csv"
    df_sweep.to_csv(sweep_csv, index=False)
    print(f"\nSaved cascade sweep results to: {sweep_csv}")

    # ============================================================
    # VISUALIZATIONS: (1) Adoption Trajectory Over Time, (2) Final Adoption vs Seed Fraction
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Plot 1: Cascade Trajectories Over Time for Representative Seed Fractions
    ax1 = axes[0]
    sample_fractions = [0.02, 0.06, 0.10, 0.15]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, f in enumerate(sample_fractions):
        traj_p = trajectories_pr.get(f, [])
        traj_r = trajectories_rnd.get(f, [])
        ax1.plot(range(len(traj_p)), [x / N * 100 for x in traj_p], marker="o", color=colors[idx], linewidth=2, label=f"Top-PR ({f*100:.0f}%)")
        ax1.plot(range(len(traj_r)), [x / N * 100 for x in traj_r], marker="s", linestyle="--", color=colors[idx], alpha=0.6, label=f"Random ({f*100:.0f}%)")

    ax1.set_title("Adoption Trajectory Over Discrete Time Steps", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time Step (t)", fontsize=11)
    ax1.set_ylabel("Cumulative Adoption (% of Network)", fontsize=11)
    ax1.legend(loc="lower right", frameon=True, fontsize=8, ncol=2)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Final Adoption vs Seed Fraction (Tipping-Point Phase Transition)
    ax2 = axes[1]
    ax2.plot(df_sweep["Seed_Fraction"] * 100, df_sweep["Top_PageRank_Final_Adoption"] * 100, marker="o", linewidth=2.5, color="#1f77b4", label="Top-PageRank Seeds (Influentials)")
    ax2.errorbar(
        df_sweep["Seed_Fraction"] * 100,
        df_sweep["Random_Final_Adoption_Mean"] * 100,
        yerr=df_sweep["Random_Final_Adoption_Std"] * 100,
        marker="s", linestyle="--", linewidth=2, color="#d62728", capsize=4, label="Random Seeds (Mean +/- Std)"
    )

    ax2.set_title("Final Adoption vs Seed Size (Tipping-Point Transition)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Initial Seed Size (% of Network)", fontsize=11)
    ax2.set_ylabel("Final Network Adoption (% of Network)", fontsize=11)
    ax2.legend(loc="lower right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Granovetter Threshold Diffusion Dynamics on Real Network", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = OUTPUTS_DIR / f"{dataset_key}_fig6_granovetter_cascade_dynamics.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved Granovetter cascade publication figure to: {fig_path}")

    return df_sweep

if __name__ == "__main__":
    simulate_real_network_cascades()
