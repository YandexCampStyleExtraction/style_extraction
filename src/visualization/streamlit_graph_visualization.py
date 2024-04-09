import numpy as np
from scipy.spatial import Delaunay
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# Function to generate random embeddings
def generate_random_embeddings(num_points=100, emb_dim=2):
    emb = np.random.uniform(-1, 1, size=(num_points, emb_dim))
    labels = np.arange(num_points)
    np.random.shuffle(labels)
    return emb, labels

# Function to compute Gabriel Graph
def compute_gabriel_graph(embeddings):
    tri = Delaunay(embeddings)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                edges.add((simplex[i], simplex[j]))
    gabriel_edges = set()
    for i, j in edges:
        midpoint = (embeddings[i] + embeddings[j]) / 2
        radius = np.linalg.norm(embeddings[i] - embeddings[j]) / 2
        is_gabriel_edge = True
        for k in range(len(embeddings)):
            if k != i and k != j:
                dist_to_midpoint = np.linalg.norm(midpoint - embeddings[k])
                if dist_to_midpoint < radius:
                    is_gabriel_edge = False
                    break
        if is_gabriel_edge:
            gabriel_edges.add((i, j))
    return gabriel_edges

# Streamlit app
def main():
    st.title("Gabriel Graph Visualization")

    # Generate random embeddings
    num_points = st.slider("Number of points", min_value=10, max_value=200, value=100, step=10)
    embeddings, labels = generate_random_embeddings(num_points=num_points)
    print(labels)

    # Compute Gabriel Graph
    gabriel_edges = compute_gabriel_graph(embeddings)

    # Create nodes and edges for visualization
    nodes = [Node(id=str(labels[i]), size=20) for i in range(num_points)]
    edges = [Edge(str(i), str(j)) for i, j in gabriel_edges]

    # Visualization config
    config = Config(width=1000, height=1000, nodeHighlightBehavior=True, highlightColor="#F7A7A6", directed=False,
                    collapsible=True, node={'labelProperty': 'id'}, link={'highlightColor': '#8B008B'})

    # Render the graph using agraph
    st.subheader("Gabriel Graph Visualization")
    agraph(nodes=nodes, edges=edges, config=config)

if __name__ == "__main__":
    main()
