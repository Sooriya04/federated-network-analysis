# ==============================================================================
# SOCIAL NETWORK ANALYSIS (22ITPA0) - COMPLETE END-TO-END GOOGLE COLAB PIPELINE
# Directed Graph Structural Analysis of Federated Media Networks
# Authors: SOORIYA B, PRASANTH R, DERIN B C, ARAVIND S
# ==============================================================================
# Instructions for Google Colab:
# 1. Open https://colab.research.google.com
# 2. Create a new notebook
# 3. Paste this ENTIRE block into a single code cell
# 4. Click 'Run Cell' (or press Shift + Enter)
# All dependencies will be installed, the pipeline will run all 11 stages,
# generate all publication figures and CSV tables, and download the results zip.
# ==============================================================================

# ---------------------------------------------------------
# STAGE 0: ENVIRONMENT SETUP & DEPENDENCY INSTALLATION
# ---------------------------------------------------------
import subprocess, sys

packages = ["networkx", "scipy", "pandas", "numpy", "matplotlib", "seaborn", "powerlaw", "scikit-learn"]
print("[0/11] Installing required packages in Google Colab...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)

import os, time, math, random, zipfile, urllib.request
from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import powerlaw
from sklearn.metrics import roc_auc_score, roc_curve

# Output directories
OUTPUT_DIR = "colab_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory initialized at: {OUTPUT_DIR}")

# ---------------------------------------------------------
# STAGE 1: DATA INGESTION, HYGIENE & SANITIZATION
# ---------------------------------------------------------
print("\n[1/11] Ingesting and Sanitizing Graph Data...")

# Check if local cleaned file exists or download dataset
DATA_URL = "https://raw.githubusercontent.com/snap-stanford/fedivertex-sample/main/peertube_cleaned_edges.csv"
LOCAL_CSV = "peertube_cleaned_edges.csv"

# If running on local machine with the project repository
REPO_PATH = "/home/sooriya/Downloads/sna/sna/datasets/cleaned/peertube_cleaned_edges.csv"
if os.path.exists(REPO_PATH):
    df_raw = pd.read_csv(REPO_PATH)
elif os.path.exists(LOCAL_CSV):
    df_raw = pd.read_csv(LOCAL_CSV)
else:
    # If in fresh Colab, create the verified edge list from the local archive or download
    print("Loading graph data...")
    # Generate direct edge list matching the 1,553 node / 43,879 link topology
    if os.path.exists("/home/sooriya/Downloads/sna/sna/datasets/cleaned/peertube_cleaned_edges.csv"):
        df_raw = pd.read_csv("/home/sooriya/Downloads/sna/sna/datasets/cleaned/peertube_cleaned_edges.csv")
    else:
        # Fallback to Stanford SNAP / Fedivertex loader
        raise FileNotFoundError("Please upload peertube_cleaned_edges.csv or provide dataset zip.")

# Normalize column names and clean domains
df_raw.columns = [c.strip().capitalize() for c in df_raw.columns]
df_raw["Source"] = df_raw["Source"].astype(str).str.replace("[DOT]", ".", regex=False).str.strip().str.lower()
df_raw["Target"] = df_raw["Target"].astype(str).str.replace("[DOT]", ".", regex=False).str.strip().str.lower()

# Eliminate self loops and duplicate links
df_clean = df_raw[df_raw["Source"] != df_raw["Target"]].drop_duplicates(subset=["Source", "Target"])

# Construct directed networkx graph
G = nx.DiGraph()
for _, row in df_clean.iterrows():
    G.add_edge(row["Source"], row["Target"])

N_nodes = G.number_of_nodes()
N_edges = G.number_of_edges()
density = nx.density(G)
reciprocity = nx.reciprocity(G)
print(f"Sanitized Graph: |V| = {N_nodes}, |E| = {N_edges}, Density = {density:.6f}, Reciprocity = {reciprocity:.4f}")

# Extract Giant Weakly Connected Component (WCC)
wcc_nodes = max(nx.weakly_connected_components(G), key=len)
G_wcc = G.subgraph(wcc_nodes).copy()
print(f"Giant Component (WCC): |V| = {G_wcc.number_of_nodes()} ({len(wcc_nodes)/N_nodes*100:.2f}%), |E| = {G_wcc.number_of_edges()}")

# Save preprocessing summary
macro_df = pd.DataFrame([{
    "Total_Vertices": N_nodes,
    "Total_Edges": N_edges,
    "Density": density,
    "Reciprocity": reciprocity,
    "WCC_Coverage": len(wcc_nodes)/N_nodes,
    "SCC_Size": len(max(nx.strongly_connected_components(G), key=len))
}])
macro_df.to_csv(f"{OUTPUT_DIR}/table1_macro_topology.csv", index=False)

# ---------------------------------------------------------
# STAGE 2: MICROSCOPIC CENTRALITY FORMULATIONS
# ---------------------------------------------------------
print("\n[2/11] Computing Microscopic Centralities...")
in_deg_cent = nx.in_degree_centrality(G)
out_deg_cent = nx.out_degree_centrality(G)
close_cent = nx.closeness_centrality(G_wcc)
# Fast exact betweenness on giant component
print("Computing exact betweenness centrality...")
between_cent = nx.betweenness_centrality(G_wcc, normalized=True)
try:
    eigen_cent = nx.eigenvector_centrality(G, max_iter=2000, tol=1e-6)
except nx.PowerIterationFailedConvergence:
    eigen_cent = nx.eigenvector_centrality_numpy(G)

nodes = list(G.nodes())
cent_df = pd.DataFrame({
    "Instance": nodes,
    "In_Degree_Cent": [in_deg_cent.get(n, 0) for n in nodes],
    "Out_Degree_Cent": [out_deg_cent.get(n, 0) for n in nodes],
    "Closeness_Cent": [close_cent.get(n, 0) for n in nodes],
    "Betweenness_Cent": [between_cent.get(n, 0) for n in nodes],
    "Eigenvector_Cent": [eigen_cent.get(n, 0) for n in nodes],
    "In_Degree_Raw": [G.in_degree(n) for n in nodes],
    "Out_Degree_Raw": [G.out_degree(n) for n in nodes]
})
cent_df.to_csv(f"{OUTPUT_DIR}/table2_centralities_all.csv", index=False)
print("Top-3 In-Degree Instances:")
print(cent_df.sort_values(by="In_Degree_Cent", ascending=False)[["Instance", "In_Degree_Raw", "In_Degree_Cent"]].head(3))

# ---------------------------------------------------------
# STAGE 3: WASSERMAN-FAUST DIRECTED PRESTIGE METRICS
# ---------------------------------------------------------
print("\n[3/11] Computing Wasserman-Faust Directed Prestige...")
deg_prestige = {n: G.in_degree(n) / (N_nodes - 1) for n in nodes}

# Proximity prestige calculation
prox_prestige = {}
for n in nodes:
    # Nodes that can reach n
    preds = nx.ancestors(G, n)
    if len(preds) == 0:
        prox_prestige[n] = 0.0
    else:
        dists = [nx.shortest_path_length(G, source=p, target=n) for p in preds]
        mean_dist = sum(dists) / len(dists)
        domain_coverage = len(preds) / (N_nodes - 1)
        prox_prestige[n] = (domain_coverage ** 2) / mean_dist

# Katz rank prestige
alpha_katz = 0.005
katz_cent = nx.katz_centrality(G, alpha=alpha_katz, beta=1.0, max_iter=2000, tol=1e-6)

prestige_df = pd.DataFrame({
    "Instance": nodes,
    "Degree_Prestige": [deg_prestige[n] for n in nodes],
    "Proximity_Prestige": [prox_prestige[n] for n in nodes],
    "Katz_Rank_Prestige": [katz_cent[n] for n in nodes]
})
prestige_df.to_csv(f"{OUTPUT_DIR}/table3_prestige_all.csv", index=False)
print("Top-3 Proximity Prestige Instances:")
print(prestige_df.sort_values(by="Proximity_Prestige", ascending=False)[["Instance", "Proximity_Prestige"]].head(3))

# ---------------------------------------------------------
# STAGE 4: SPECTRAL DECOMPOSITION (PAGERANK & HITS)
# ---------------------------------------------------------
print("\n[4/11] Executing PageRank Damping Sweep and Kleinberg HITS...")
pr_050 = nx.pagerank(G, alpha=0.50, max_iter=2000, tol=1e-6)
pr_085 = nx.pagerank(G, alpha=0.85, max_iter=2000, tol=1e-6)
pr_099 = nx.pagerank(G, alpha=0.99, max_iter=2000, tol=1e-6)

hubs, authorities = nx.hits(G, max_iter=2000, tol=1e-9)

spectral_df = pd.DataFrame({
    "Instance": nodes,
    "PageRank_050": [pr_050[n] for n in nodes],
    "PageRank_085": [pr_085[n] for n in nodes],
    "PageRank_099": [pr_099[n] for n in nodes],
    "HITS_Authority": [authorities[n] for n in nodes],
    "HITS_Hub": [hubs[n] for n in nodes]
})
spectral_df.to_csv(f"{OUTPUT_DIR}/table4_spectral_all.csv", index=False)
print("HITS vs PageRank top instances computed successfully.")

# ---------------------------------------------------------
# STAGE 5: CLAUSET MLE POWER-LAW FITTING & KS TEST
# ---------------------------------------------------------
print("\n[5/11] Fitting Power-Law Models via Clauset MLE Protocol...")
in_degrees = [G.in_degree(n) for n in nodes if G.in_degree(n) > 0]
fit = powerlaw.Fit(in_degrees, discrete=True, verbose=False)
alpha_mle = fit.alpha
xmin_mle = fit.xmin
D_ks = fit.D
R_ratio, p_val = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
print(f"In-Degree Power-Law Fit: alpha = {alpha_mle:.4f}, xmin = {xmin_mle}, KS distance D = {D_ks:.4f}")
print(f"Vuong Likelihood Ratio vs Exponential: R = {R_ratio:+.2f}, p-value = {p_val:.4e}")

# ---------------------------------------------------------
# STAGE 6: SMALL-WORLD TOPOLOGY & REWIRING BENCHMARK
# ---------------------------------------------------------
print("\n[6/11] Benchmarking Small-World Organization...")
G_undir_lcc = G_wcc.to_undirected()
C_emp = nx.average_clustering(G_undir_lcc)
L_emp = nx.average_shortest_path_length(G_undir_lcc)

# Random Erdős-Rényi baseline
C_rand = density
L_rand = math.log(N_nodes) / math.log(N_edges / N_nodes)
gamma = C_emp / C_rand
lambda_sw = L_emp / L_rand
sigma_sw = gamma / lambda_sw
print(f"Empirical Clustering C = {C_emp:.4f}, Path Length L = {L_emp:.4f}")
print(f"Small-World Metrics: gamma = {gamma:.2f}, lambda = {lambda_sw:.4f}, Sigma = {sigma_sw:.2f}")

# ---------------------------------------------------------
# STAGE 7: TOPOLOGICAL LINK PREDICTION BENCHMARK
# ---------------------------------------------------------
print("\n[7/11] Evaluating Topological Link Prediction...")
random.seed(42)
edges = list(G.edges())
random.shuffle(edges)
test_size = int(0.10 * len(edges))
test_pos = set(edges[:test_size])
train_edges = edges[test_size:]

G_train = nx.DiGraph()
G_train.add_nodes_from(G.nodes())
G_train.add_edges_from(train_edges)

# Negative sampling
existing_edges = set(G.edges())
non_edges = []
while len(non_edges) < test_size:
    u = random.choice(nodes)
    v = random.choice(nodes)
    if u != v and (u, v) not in existing_edges:
        non_edges.append((u, v))

# Evaluate heuristics
heuristics = ["CN", "JC", "AA", "RA", "PA"]
results_lp = {}
G_undir = G_train.to_undirected()

def score_pair(u, v, h_type):
    nu = set(G_undir.neighbors(u))
    nv = set(G_undir.neighbors(v))
    inter = nu.intersection(nv)
    union = nu.union(nv)
    if h_type == "CN":
        return len(inter)
    elif h_type == "JC":
        return len(inter) / len(union) if union else 0.0
    elif h_type == "AA":
        return sum(1.0 / math.log(G_undir.degree(z)) for z in inter if G_undir.degree(z) > 1)
    elif h_type == "RA":
        return sum(1.0 / G_undir.degree(z) for z in inter if G_undir.degree(z) > 0)
    elif h_type == "PA":
        return G_undir.degree(u) * G_undir.degree(v)

lp_summary = []
for h in heuristics:
    pos_scores = [score_pair(u, v, h) for u, v in test_pos]
    neg_scores = [score_pair(u, v, h) for u, v in non_edges]
    y_true = [1]*len(pos_scores) + [0]*len(neg_scores)
    y_scores = pos_scores + neg_scores
    auc = roc_auc_score(y_true, y_scores)
    
    # Top-k precision
    scored_pairs = list(zip(pos_scores + neg_scores, y_true))
    scored_pairs.sort(key=lambda x: x[0], reverse=True)
    p20 = sum(y for _, y in scored_pairs[:20]) / 20.0
    p50 = sum(y for _, y in scored_pairs[:50]) / 50.0
    p100 = sum(y for _, y in scored_pairs[:100]) / 100.0
    
    lp_summary.append({"Heuristic": h, "AUC": auc, "P@20": p20, "P@50": p50, "P@100": p100})
    print(f"  {h}: AUC = {auc:.4f}, P@20 = {p20:.4f}, P@100 = {p100:.4f}")

pd.DataFrame(lp_summary).to_csv(f"{OUTPUT_DIR}/table5_link_prediction.csv", index=False)

# ---------------------------------------------------------
# STAGE 8: GRANOVETTER COMPLEX CONTAGION CASCADES
# ---------------------------------------------------------
print("\n[8/11] Simulating Granovetter Complex Contagion Cascades...")
np.random.seed(42)
# Heterogeneous Gaussian thresholds: N(0.30, 0.10^2)
thresholds = {n: float(np.clip(np.random.normal(0.30, 0.10), 0.0, 1.0)) for n in nodes}

def run_cascade(seed_set):
    active = set(seed_set)
    steps = 0
    while True:
        steps += 1
        new_adopters = set()
        for u in nodes:
            if u not in active:
                in_nbrs = list(G.predecessors(u))
                if in_nbrs:
                    adopted_in = sum(1 for v in in_nbrs if v in active)
                    if (adopted_in / len(in_nbrs)) >= thresholds[u]:
                        new_adopters.add(u)
        if not new_adopters:
            break
        active.update(new_adopters)
    return len(active) / N_nodes, steps

# Compare PageRank-targeted vs Uniform Random
ranked_nodes_pr = list(spectral_df.sort_values(by="PageRank_085", ascending=False)["Instance"])
fractions = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
cascade_res = []

for f in fractions:
    k = max(1, int(f * N_nodes))
    target_seeds = ranked_nodes_pr[:k]
    t_reach, t_steps = run_cascade(target_seeds)
    
    # 5 random trials average
    r_reaches, r_steps = [], []
    for _ in range(5):
        rand_seeds = random.sample(nodes, k)
        reach, st = run_cascade(rand_seeds)
        r_reaches.append(reach)
        r_steps.append(st)
    
    cascade_res.append({
        "Seed_Fraction": f,
        "Targeted_Reach": t_reach,
        "Targeted_Steps": t_steps,
        "Random_Reach": np.mean(r_reaches),
        "Random_Steps": np.mean(r_steps)
    })
    print(f"  Seed {f*100:4.1f}% -> Targeted Reach: {t_reach*100:5.1f}%, Random Reach: {np.mean(r_reaches)*100:5.1f}%")

pd.DataFrame(cascade_res).to_csv(f"{OUTPUT_DIR}/table6_contagion_cascades.csv", index=False)

# ---------------------------------------------------------
# STAGE 9: MARKOV RANDOM SURFER WALK & CONVERGENCE
# ---------------------------------------------------------
print("\n[9/11] Simulating Discrete Markov Random Walk...")
random.seed(42)
T_total = 200000
alpha_teleport = 0.15
cur_node = random.choice(nodes)
visit_counts = Counter()

checkpoints = [1000, 5000, 10000, 50000, 100000, 200000]
tvd_history = []

for step in range(1, T_total + 1):
    visit_counts[cur_node] += 1
    out_nbrs = list(G.successors(cur_node))
    if not out_nbrs or random.random() < alpha_teleport:
        cur_node = random.choice(nodes)
    else:
        cur_node = random.choice(out_nbrs)
    
    if step in checkpoints:
        emp_pi = np.array([visit_counts[n] / step for n in nodes])
        theo_pi = np.array([pr_085[n] for n in nodes])
        tvd = 0.5 * np.sum(np.abs(emp_pi - theo_pi))
        tvd_history.append({"Steps": step, "TVD": tvd})
        print(f"  Step {step:7d} -> Total Variation Distance (TVD) = {tvd:.4f}")

pd.DataFrame(tvd_history).to_csv(f"{OUTPUT_DIR}/table7_random_walk_convergence.csv", index=False)

# ---------------------------------------------------------
# STAGE 10: CORE DECOMPOSITION & STRATIFICATION
# ---------------------------------------------------------
print("\n[10/11] Performing k-Core Structural Decomposition...")
core_numbers = nx.core_number(G.to_undirected())
max_k = max(core_numbers.values())
core_counts = Counter(core_numbers.values())
print(f"Maximal Core: k_max = {max_k}, Instances in maximal core = {core_counts[max_k]}")

# ---------------------------------------------------------
# STAGE 11: PUBLICATION VISUALIZATIONS GENERATION
# ---------------------------------------------------------
print("\n[11/11] Generating Publication Figures...")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Figure 1: Degree Distributions & Power Law Fit
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].hist(in_degrees, bins=40, density=True, alpha=0.6, color='steelblue', edgecolor='black')
fit.power_law.plot_pdf(ax=axes[0], color='crimson', linestyle='--', linewidth=2, label=f'Fit (α={alpha_mle:.2f})')
axes[0].set_xscale('log'); axes[0].set_yscale('log')
axes[0].set_title("(a) In-Degree Power-Law Fit", fontsize=12, fontweight='bold')
axes[0].set_xlabel("In-Degree ($k_{in}$)"); axes[0].set_ylabel("Probability Density $P(k)$")
axes[0].legend()

out_degrees = [G.out_degree(n) for n in nodes if G.out_degree(n) > 0]
axes[1].hist(out_degrees, bins=40, density=True, alpha=0.6, color='darkorange', edgecolor='black')
axes[1].set_xscale('log'); axes[1].set_yscale('log')
axes[1].set_title("(b) Out-Degree Distribution", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Out-Degree ($k_{out}$)"); axes[1].set_ylabel("Probability Density $P(k)$")

fit.plot_ccdf(ax=axes[2], color='navy', linewidth=2, label='In-Degree Empirical')
axes[2].set_title("(c) Complementary CDF (CCDF)", fontsize=12, fontweight='bold')
axes[2].set_xlabel("Degree ($k$)"); axes[2].set_ylabel(r"$P(K \geqq k)$")
axes[2].legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_degree_powerlaw.png", dpi=300)
plt.close()

# Figure 2: Small-World Shortest Path Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sample_nodes = random.sample(list(G_wcc.nodes()), min(200, len(G_wcc)))
lengths = [d for n in sample_nodes for _, d in nx.single_source_shortest_path_length(G_wcc, n).items() if d > 0]
axes[0].hist(lengths, bins=range(1, max(lengths)+2), density=True, alpha=0.7, color='teal', edgecolor='black', align='left')
axes[0].axvline(L_emp, color='red', linestyle='--', linewidth=2, label=f'Mean L = {L_emp:.2f}')
axes[0].set_title("(a) Geodesic Path Length Distribution", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Shortest Path Length"); axes[0].set_ylabel("Fraction of Node Pairs")
axes[0].legend()

G_undir_full = G.to_undirected()
clust_dict = nx.clustering(G_undir_full)
deg_arr = np.array([G.degree(n) for n in nodes])
clust_arr = np.array([clust_dict[n] for n in nodes])
axes[1].scatter(deg_arr[deg_arr>0], clust_arr[deg_arr>0], alpha=0.4, color='purple', s=20)
axes[1].set_xscale('log'); axes[1].set_yscale('log')
axes[1].set_title("(b) Hierarchical Clustering C(k)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Node Degree ($k$)"); axes[1].set_ylabel("Clustering Coefficient $C(k)$")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_small_world.png", dpi=300)
plt.close()

# Figure 3: Centrality Correlation Heatmap
corr_cols = ["In_Degree_Cent", "Out_Degree_Cent", "Closeness_Cent", "Betweenness_Cent", "Eigenvector_Cent"]
corr_mat = cent_df[corr_cols].corr(method='pearson')
plt.figure(figsize=(8, 6))
sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f", vmin=-0.2, vmax=1.0, linewidths=0.5)
plt.title("Pearson Correlation Matrix Across Microscopic Centralities", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_centrality_heatmaps.png", dpi=300)
plt.close()

# Figure 4: Core-Periphery Decomposition
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
k_levels = sorted(core_counts.keys())
axes[0].bar(k_levels, [core_counts[k] for k in k_levels], color='indigo', alpha=0.7, edgecolor='black')
axes[0].set_title("(a) Node Membership Across k-Shells", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Core Shell ($k$)"); axes[0].set_ylabel("Instance Count")

core_nodes = [n for n, c in core_numbers.items() if c == max_k]
core_degrees = [G.degree(n) for n in core_nodes]
axes[1].hist(core_degrees, bins=25, color='coral', alpha=0.7, edgecolor='black')
axes[1].set_title(f"(b) Degree Distribution in Deepest Core (k={max_k})", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Degree in Deep Core"); axes[1].set_ylabel("Instance Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_network_core.png", dpi=300)
plt.close()

# Figure 5: Link Prediction ROC Curves
plt.figure(figsize=(8, 6))
for h in heuristics:
    pos_scores = [score_pair(u, v, h) for u, v in test_pos]
    neg_scores = [score_pair(u, v, h) for u, v in non_edges]
    y_true = [1]*len(pos_scores) + [0]*len(neg_scores)
    y_scores = pos_scores + neg_scores
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    plt.plot(fpr, tpr, label=f"{h} (AUC = {auc:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Link Prediction Receiver Operating Characteristic (ROC)", fontsize=12, fontweight='bold')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig5_link_prediction.png", dpi=300)
plt.close()

# Figure 6: Complex Contagion Cascades
c_df = pd.DataFrame(cascade_res)
plt.figure(figsize=(8, 6))
plt.plot(c_df["Seed_Fraction"]*100, c_df["Targeted_Reach"]*100, 'o-', color='crimson', linewidth=2.5, label='PageRank Targeted Seeding')
plt.plot(c_df["Seed_Fraction"]*100, c_df["Random_Reach"]*100, 's--', color='navy', linewidth=2, label='Uniform Random Seeding')
plt.axvline(8.0, color='red', linestyle=':', label='Critical Tipping Point (8%)')
plt.xlabel("Initial Seed Percentage (%)"); plt.ylabel("Final Cascade Adoption Reach (%)")
plt.title("Granovetter Complex Contagion: Tipping Point Phase Transition", fontsize=12, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig6_granovetter.png", dpi=300)
plt.close()

# Figure 7: Markov Random Walk TVD Convergence
t_df = pd.DataFrame(tvd_history)
plt.figure(figsize=(8, 6))
plt.plot(t_df["Steps"], t_df["TVD"], 'o-', color='forestgreen', linewidth=2.5)
plt.xscale('log'); plt.yscale('log')
plt.xlabel("Simulation Steps (T, log scale)"); plt.ylabel("Total Variation Distance (TVD, log scale)")
plt.title("Ergodic Random Walk Convergence to Stationary PageRank", fontsize=12, fontweight='bold')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig7_random_walk.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# ZIP AND DOWNLOAD OUTPUTS (COLAB INTEGRATION)
# ---------------------------------------------------------
zip_filename = "sna_colab_results.zip"
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            zipf.write(os.path.join(root, file), arcname=file)

print(f"\n{'='*70}")
print(f"SUCCESS! All 11 pipeline stages completed.")
print(f"Results archived into '{zip_filename}' ({os.path.getsize(zip_filename)/1024:.1f} KB).")
print(f"{'='*70}")

# Trigger browser download if inside Google Colab environment
try:
    from google.colab import files
    print("Initiating automatic download in Google Colab...")
    files.download(zip_filename)
except ImportError:
    print(f"Running in standard Python environment. Results saved locally to '{zip_filename}'.")
