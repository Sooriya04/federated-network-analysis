"""
Gephi Lite Visualizer Export Script
Generates production-ready graph files for Gephi Lite (https://gephi.org/gephi-lite/):
  1. GEXF 1.2 with viz coordinates (x, y), node sizes, and cluster colors.
  2. Graphology JSON format for instantaneous drag-and-drop web import.
  3. Nodes and Edges CSV files with complete topological metrics.
"""

import os, sys, json, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GEPHI_DIR = PROJECT_ROOT / "gephi_lite"
GEPHI_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("GEPHI LITE ASSET GENERATOR")
print(f"Export directory: {GEPHI_DIR}")
print("="*70)

# 1. Load Sanitized Edges
edges_csv = PROJECT_ROOT / "datasets" / "cleaned" / "peertube_cleaned_edges.csv"
print(f"Loading edge list from: {edges_csv}")
df_edges = pd.read_csv(edges_csv)
df_edges.columns = [c.strip().capitalize() for c in df_edges.columns]
df_edges["Source"] = df_edges["Source"].astype(str).str.replace("[DOT]", ".", regex=False).str.strip().str.lower()
df_edges["Target"] = df_edges["Target"].astype(str).str.replace("[DOT]", ".", regex=False).str.strip().str.lower()
df_clean = df_edges[df_edges["Source"] != df_edges["Target"]].drop_duplicates(subset=["Source", "Target"])

G = nx.DiGraph()
for _, r in df_clean.iterrows():
    G.add_edge(r["Source"], r["Target"])

N = G.number_of_nodes()
M = G.number_of_edges()
print(f"Loaded directed graph: |V| = {N:,} nodes, |E| = {M:,} edges")

# 2. Load pre-computed centrality & spectral metrics
metrics_map = {}

def clean_node_name(name):
    return str(name).replace("[DOT]", ".").strip().lower()

# Centralities
cent_file = OUTPUTS_DIR / "peertube_centrality_all.csv"
if cent_file.exists():
    df_cent = pd.read_csv(cent_file)
    df_cent["node"] = df_cent["node"].apply(clean_node_name)
    for _, row in df_cent.iterrows():
        n = row["node"]
        metrics_map[n] = {
            "in_degree": int(row.get("in_degree", G.in_degree(n))),
            "out_degree": int(row.get("out_degree", G.out_degree(n))),
            "closeness": float(row.get("closeness", 0.0)),
            "betweenness": float(row.get("betweenness", 0.0)),
            "eigenvector": float(row.get("eigenvector", 0.0))
        }

# PageRank
pr_file = OUTPUTS_DIR / "peertube_pagerank_all.csv"
if pr_file.exists():
    df_pr = pd.read_csv(pr_file)
    df_pr["node"] = df_pr["node"].apply(clean_node_name)
    for _, row in df_pr.iterrows():
        n = row["node"]
        if n in metrics_map:
            metrics_map[n]["pagerank_085"] = float(row.get("pagerank_085", 0.0))
            metrics_map[n]["pagerank_050"] = float(row.get("pagerank_050", 0.0))
            metrics_map[n]["pagerank_099"] = float(row.get("pagerank_099", 0.0))

# HITS
hits_file = OUTPUTS_DIR / "peertube_hits_all.csv"
if hits_file.exists():
    df_hits = pd.read_csv(hits_file)
    df_hits["node"] = df_hits["node"].apply(clean_node_name)
    for _, row in df_hits.iterrows():
        n = row["node"]
        if n in metrics_map:
            metrics_map[n]["authority_score"] = float(row.get("authority_score", 0.0))
            metrics_map[n]["hub_score"] = float(row.get("hub_score", 0.0))

# Fallback for any missing nodes
for n in G.nodes():
    if n not in metrics_map:
        metrics_map[n] = {
            "in_degree": G.in_degree(n),
            "out_degree": G.out_degree(n),
            "closeness": 0.0,
            "betweenness": 0.0,
            "eigenvector": 0.0,
            "pagerank_085": 1.0 / N,
            "pagerank_050": 1.0 / N,
            "pagerank_099": 1.0 / N,
            "authority_score": 0.0,
            "hub_score": 0.0
        }

# 3. Community Detection & Core Numbers
print("Detecting structural communities (Louvain)...")
G_undir = G.to_undirected()
communities = nx.community.louvain_communities(G_undir, seed=42)
comm_map = {}
for cid, comm in enumerate(communities):
    for node in comm:
        comm_map[node] = cid
print(f"Identified {len(communities)} modular communities.")

core_numbers = nx.core_number(G_undir)

# 4. Color Palette Mapping
palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"
]

def hex_to_rgb(hex_str):
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# 5. Force-Directed Layout Computation
print("Computing 2D Force-Directed Layout (Spring Embedding)...")
# Seeded reproducible layout scaled to [-800, 800]
pos_raw = nx.spring_layout(G_undir, k=0.15, iterations=60, seed=42)
scale = 800.0
pos = {n: (coords[0] * scale, coords[1] * scale) for n, coords in pos_raw.items()}
print("Layout coordinates successfully calculated.")

# 6. Sizing Calculation (Log-scaled PageRank + In-Degree)
pr_values = [metrics_map[n].get("pagerank_085", 1.0/N) for n in G.nodes()]
min_pr, max_pr = min(pr_values), max(pr_values)

def calculate_node_size(n):
    pr = metrics_map[n].get("pagerank_085", 1.0/N)
    in_d = metrics_map[n].get("in_degree", 0)
    # Log-proportional radius from 3.0 to 45.0
    norm_val = (math.log(pr + 1e-6) - math.log(min_pr + 1e-6)) / (math.log(max_pr + 1e-6) - math.log(min_pr + 1e-6) + 1e-9)
    norm_val = max(0.0, min(1.0, norm_val))
    return round(3.5 + 40.0 * (norm_val ** 1.3), 2)

# 7. Construct Enhanced NetworkX Graph with Viz Metadata
G_viz = nx.DiGraph()

nodes_table = []
for n in G.nodes():
    cid = comm_map.get(n, 0)
    hex_col = palette[cid % len(palette)]
    r, g, b = hex_to_rgb(hex_col)
    size = calculate_node_size(n)
    x, y = pos[n]
    m = metrics_map[n]
    
    node_attrs = {
        "label": n,
        "community": cid,
        "k_core": core_numbers.get(n, 1),
        "in_degree": m["in_degree"],
        "out_degree": m["out_degree"],
        "closeness": round(m.get("closeness", 0.0), 6),
        "betweenness": round(m.get("betweenness", 0.0), 6),
        "eigenvector": round(m.get("eigenvector", 0.0), 6),
        "pagerank": round(m.get("pagerank_085", 0.0), 6),
        "authority": round(m.get("authority_score", 0.0), 6),
        "hub": round(m.get("hub_score", 0.0), 6),
        "viz": {
            "position": {"x": round(x, 2), "y": round(y, 2), "z": 0.0},
            "color": {"r": r, "g": g, "b": b, "a": 1.0},
            "size": size
        }
    }
    G_viz.add_node(n, **node_attrs)
    
    nodes_table.append({
        "Id": n,
        "Label": n,
        "x": round(x, 2),
        "y": round(y, 2),
        "size": size,
        "color": hex_col,
        "Community": cid,
        "K_Core": core_numbers.get(n, 1),
        "InDegree": m["in_degree"],
        "OutDegree": m["out_degree"],
        "PageRank": round(m.get("pagerank_085", 0.0), 6),
        "Betweenness": round(m.get("betweenness", 0.0), 6),
        "Closeness": round(m.get("closeness", 0.0), 6),
        "Authority": round(m.get("authority_score", 0.0), 6),
        "Hub": round(m.get("hub_score", 0.0), 6)
    })

# Add edges with weights
edges_table = []
for u, v in G.edges():
    G_viz.add_edge(u, v, weight=1.0)
    edges_table.append({
        "Source": u,
        "Target": v,
        "Type": "Directed",
        "Weight": 1.0
    })

# 8. Export Files
print("\nExporting files for Gephi Lite...")

# A. GEXF File
gexf_path = GEPHI_DIR / "peertube_gephi_lite.gexf"
nx.write_gexf(G_viz, gexf_path, version="1.2draft")
print(f"  [1/4] GEXF 1.2 with viz coordinates: {gexf_path} ({gexf_path.stat().st_size / 1024:.1f} KB)")

# B. Graphology JSON format (Native Gephi Lite Format)
json_data = {
    "attributes": {
        "name": "PeerTube Federated Media Network",
        "description": "ActivityPub directed instance subscription network",
        "node_count": N,
        "edge_count": M
    },
    "options": {
        "type": "directed",
        "multi": False
    },
    "nodes": [
        {
            "key": row["Id"],
            "attributes": {
                "label": row["Label"],
                "x": row["x"],
                "y": row["y"],
                "size": row["size"],
                "color": row["color"],
                "community": row["Community"],
                "inDegree": row["InDegree"],
                "outDegree": row["OutDegree"],
                "pagerank": row["PageRank"],
                "betweenness": row["Betweenness"],
                "authority": row["Authority"],
                "hub": row["Hub"]
            }
        }
        for row in nodes_table
    ],
    "edges": [
        {
            "source": row["Source"],
            "target": row["Target"],
            "attributes": {
                "weight": row["Weight"]
            }
        }
        for row in edges_table
    ]
}

json_path = GEPHI_DIR / "peertube_gephi_lite.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2)
print(f"  [2/4] Graphology JSON (Instant Drag & Drop): {json_path} ({json_path.stat().st_size / 1024 / 1024:.2f} MB)")

# C. Nodes CSV
nodes_csv_path = GEPHI_DIR / "gephi_nodes.csv"
pd.DataFrame(nodes_table).to_csv(nodes_csv_path, index=False)
print(f"  [3/4] Nodes CSV: {nodes_csv_path} ({nodes_csv_path.stat().st_size / 1024:.1f} KB)")

# D. Edges CSV
edges_csv_path = GEPHI_DIR / "gephi_edges.csv"
pd.DataFrame(edges_table).to_csv(edges_csv_path, index=False)
print(f"  [4/4] Edges CSV: {edges_csv_path} ({edges_csv_path.stat().st_size / 1024 / 1024:.2f} MB)")

# 9. Create a handy ZIP archive for Gephi Lite
zip_path = GEPHI_DIR / "gephi_lite_package.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(gexf_path, arcname="peertube_gephi_lite.gexf")
    zf.write(json_path, arcname="peertube_gephi_lite.json")
    zf.write(nodes_csv_path, arcname="gephi_nodes.csv")
    zf.write(edges_csv_path, arcname="gephi_edges.csv")
print(f"  [Package] Consolidated ZIP: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")

print("="*70)
print("SUCCESS: All Gephi Lite visualization assets generated!")
print("="*70)
