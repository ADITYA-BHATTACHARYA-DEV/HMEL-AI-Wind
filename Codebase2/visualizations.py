# File 5: visualization.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

# ------------------------------------------------------
# 1. Time‑Series Wind Direction Plot
# ------------------------------------------------------
def plot_direction_over_time(df):
    plt.figure(figsize=(10,4))
    plt.plot(df['Timestamp'], df['Value'])
    plt.title('Wind Direction Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Direction (degrees)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 2. Wind Rose / Polar Plot
# ------------------------------------------------------
def plot_wind_rose(df):
    angles = np.deg2rad(df['Value'])
    counts, bins = np.histogram(df['Value'], bins=16, range=(0,360))
    widths = np.deg2rad(np.diff(bins))

    ax = plt.subplot(111, polar=True)
    ax.bar(np.deg2rad(bins[:-1]), counts, width=widths)
    ax.set_title('Wind Rose (Direction Only)')
    plt.show()

# ------------------------------------------------------
# 3. Training Loss Curve
# ------------------------------------------------------
def plot_training_curve(loss_list):
    plt.figure(figsize=(6,4))
    plt.plot(loss_list, marker='o')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 4. Predictions vs Actual
# ------------------------------------------------------
def plot_predictions(true_vals, predictions):
    plt.figure(figsize=(6,4))
    plt.plot(true_vals, label='True')
    plt.plot(predictions, label='Predicted')
    plt.title('Model Predictions vs True')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 5. GNN Graph Visualization
# ------------------------------------------------------
def visualize_graph(x_nodes, edge_index):
    G = nx.Graph()
    for i in range(x_nodes.shape[0]):
        G.add_node(i)
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    nx.draw(G, with_labels=True, node_color='lightblue')
    plt.title('GNN Graph Structure')
    plt.show()

# ------------------------------------------------------
# 6. Vertical Wind Profiler Visualization
# ------------------------------------------------------
# This assumes you have wind direction / wind speed by height.
# df must contain columns: Height, Direction, Speed(optional)
# ------------------------------------------------------

def plot_vertical_wind_profile(df):
    if 'Height' not in df.columns or 'Value' not in df.columns:
        raise ValueError('DataFrame must contain Height and Value (direction) columns')

    heights = df['Height']
    directions = df['Value']

    plt.figure(figsize=(6,6))
    plt.plot(directions, heights, marker='o')
    plt.xlabel('Wind Direction (degrees)')
    plt.ylabel('Height (m)')
    plt.title('Vertical Wind Direction Profile')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Optional: Vertical speed profile if speed column exists

def plot_vertical_speed_profile(df):
    if 'Height' not in df.columns or 'Speed' not in df.columns:
        raise ValueError('DataFrame must contain Height and Speed columns')

    plt.figure(figsize=(6,6))
    plt.plot(df['Speed'], df['Height'], marker='o', color='red')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Height (m)')
    plt.title('Vertical Wind Speed Profile')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
