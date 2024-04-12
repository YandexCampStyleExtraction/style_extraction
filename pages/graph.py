import numpy as np
from scipy.spatial import Delaunay
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from collections import defaultdict
import pandas as pd
import colorsys
from sklearn.decomposition import PCA

import ast  # Add this import

# Function to compute Gabriel Graph
def compute_gabriel_graph(embeddings):
    tri = Delaunay(embeddings)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
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

# Generate label color map with equally spaced colors
def generate_label_color_map(labels):
    color_map = defaultdict(list)
    for label in labels:
        for i in range(100):
            hue = i / 100.0
            saturation = 0.5
            value = 0.8
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            color_map[label].append(f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})")
    return color_map


def plotting_graph():
    st.title("Gabriel Graph Visualization")

    # Read data from CSV file
    embds = np.loadtxt('data_visualisation/embeddings.txt')
    embds_pca = PCA(n_components=10).fit_transform(embds)

    author_years = pd.read_csv("data_visualisation/author_year.csv")

    data = pd.DataFrame({
        "emb": embds_pca.tolist(),
        "author": author_years["author"].apply(lambda x: x.split(',')[0]),
        "date": author_years["year"]
    })

    date_threshold = st.slider("Year Threshold", min_value=1700, max_value=2010, value=2010, step=10)
    num_points = st.slider("Number of points", min_value=10, max_value=100, value=30, step=5)

    # Filter data based on date threshold
    data_filtered = data[data['date'] <= date_threshold].sample(num_points, replace=True)

    #print(data['emb'].sample(10))

    # Convert 'emb' column from string to list using ast.literal_eval
    #data_filtered['emb'] = data_filtered['emb'].apply(ast.literal_eval)

    # Extract embeddings, labels, and dates
    embeddings = np.array(data_filtered['emb'].to_list())
    labels = data_filtered['author'].values

    # Compute Gabriel Graph
    gabriel_edges = compute_gabriel_graph(embeddings)

    # Create nodes and edges for visualization
    label_color_map = generate_label_color_map(labels)
    nodes = [Node(id=i, size=50, color=label_color_map[labels[i]][i % 1000], label=str(labels[i]), labelColor='#FFFFFF', shape='box') for i in range(len(labels))]
    edges = [Edge(str(i), str(j)) for i, j in gabriel_edges]

    # Visualization config
    config = Config(width=1000, height=1000, nodeHighlightBehavior=True, highlightColor="#F7A7A6", directed=False,
                    collapsible=True, node={'labelProperty': 'label'}, link={'highlightColor': '#8B008B'})

    # Render the graph using agraph
    st.subheader("Gabriel Graph Visualization")
    agraph(nodes=nodes, edges=edges, config=config)

    st.button("Re-run")


st.set_page_config(page_title="Gabriel Graph Visualization", page_icon="📈")
st.markdown("# Graph Visualization")
st.sidebar.header("Graph Visualization")
st.write(
    """Some text... Bla Bla Bla..."""
)

plotting_graph()
