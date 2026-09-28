import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# ============================================
# Locate Project Folder
# ============================================

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)

dataset_path = os.path.join(project_root, "datasets")

# ============================================
# Read Datasets
# ============================================

herbs = pd.read_csv(os.path.join(dataset_path, "herbs.csv"))
drugs = pd.read_csv(os.path.join(dataset_path, "drugs.csv"))
proteins = pd.read_csv(os.path.join(dataset_path, "proteins.csv"))
compound_protein = pd.read_csv(
    os.path.join(dataset_path, "compound_protein.csv")
)

# ============================================
# Create Knowledge Graph
# ============================================

G = nx.Graph()

# ============================================
# Herb → Compound
# ============================================

for _, row in herbs.iterrows():

    herb = row["Herb"]
    compound = row["Active_Compound"]

    G.add_node(herb, type="Herb")
    G.add_node(compound, type="Compound")

    G.add_edge(
        herb,
        compound,
        relation="contains"
    )

# ============================================
# Compound → Protein
# ============================================

for _, row in compound_protein.iterrows():

    compound = row["Compound"]
    protein = row["Protein"]

    G.add_node(compound, type="Compound")
    G.add_node(protein, type="Protein")

    G.add_edge(
        compound,
        protein,
        relation="binds_to"
    )

# ============================================
# Drug → Protein
# ============================================

for _, row in drugs.iterrows():

    drug = row["Drug"]
    protein = row["Target_Protein"]

    G.add_node(drug, type="Drug")
    G.add_node(protein, type="Protein")

    G.add_edge(
        drug,
        protein,
        relation="targets"
    )

# ============================================
# Protein → Pathway
# ============================================

for _, row in proteins.iterrows():

    protein = row["Protein"]
    pathway = row["Pathway"]

    G.add_node(pathway, type="Pathway")

    G.add_edge(
        protein,
        pathway,
        relation="belongs_to"
    )

# ============================================
# Graph Summary
# ============================================

print("\n========== KNOWLEDGE GRAPH SUMMARY ==========")
print("Number of Nodes :", G.number_of_nodes())
print("Number of Edges :", G.number_of_edges())

print("\n========== NODES ==========")
print(G.nodes(data=True))

print("\n========== EDGES ==========")
print(G.edges(data=True))

# ============================================
# Draw Graph
# ============================================

plt.figure(figsize=(16, 10))

pos = nx.spring_layout(G, seed=42)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=1200
)

nx.draw_networkx_edges(
    G,
    pos,
    width=2
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=8,
    font_weight="bold"
)

plt.title("AI Herb-Drug Knowledge Graph")
plt.axis("off")
plt.show()