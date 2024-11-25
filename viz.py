import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from common.tensor import tensor
from data.dataset import Dataset
import networkx as nx
# from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def plot_samples(x, y, title='Samples'):
    plt.figure()
    plt.scatter(x,y,color='dodgerblue', edgecolor='k', alpha=0.7)
    plt.title(title)

def plot_loss_curve(losses_list, legends=None, title='Training Loss Curve'):
    """
    Plot the loss curve for multiple datasets over epochs.

    Args:
    losses_list: List of lists, where each inner list represents losses over epochs.
    legends: List of strings representing the legend for each curve.
    title: Title of the plot (default: 'Training Loss Curve').
    """
    plt.figure(figsize=(10, 6))
    
    if legends is None:
        legends = [f'Loss {i+1}' for i in range(len(losses_list))]
    
    for losses, legend in zip(losses_list, legends):
        plt.plot(losses, linewidth=2, label=legend)
    
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True)
    
    plt.legend()

from sklearn.decomposition import PCA
def pca_and_visualize(features, labels, n_components=2, visualize=True, save_path=None, alpha_dict=None):
    """
    Perform PCA on the features and visualize the results.

    Parameters:
    features (np.ndarray): The feature matrix to be reduced.
    labels (np.ndarray): The labels corresponding to the data points, used for color-coding.
    n_components (int): Number of dimensions for PCA (2 or 3). Default is 2.
    visualize (bool): Whether to display the plot interactively. Default is True.
    save_path (str): Path to save the plot as an image. If None, the plot will not be saved. Default is None.
    alpha_dict (dict): Dictionary mapping labels to alpha values for transparency. Default is None.

    Returns:
    None
    """
    # Ensure the number of components is either 2 or 3
    if n_components not in [2, 3]:
        raise ValueError("n_components must be either 2 or 3.")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    
    # Visualization
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(labels)
        for label in unique_labels:
            alpha = alpha_dict.get(label, 0.5) if alpha_dict else 0.5
            plt.scatter(reduced_features[labels == label, 0], reduced_features[labels == label, 1], 
                        label=f'Label {label}', alpha=alpha)
        plt.colorbar(label='Labels')
        plt.title('PCA (2D)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if visualize:
            plt.show()
        plt.close()
    
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        unique_labels = np.unique(labels)
        for label in unique_labels:
            alpha = alpha_dict.get(label, 0.5) if alpha_dict else 0.5
            ax.scatter(reduced_features[labels == label, 0], reduced_features[labels == label, 1], 
                       reduced_features[labels == label, 2], label=f'Label {label}', alpha=alpha)
        fig.colorbar(ax.collections[0], ax=ax, label='Labels')
        ax.set_title('PCA (3D)')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if visualize:
            plt.show()
        plt.close()

def plot_metrics(metric_names, metric_values):
    plt.figure(figsize=(10, 6))
    
    for i, metric in enumerate(metric_names):
        plt.plot(metric_values[i], label=metric)
        
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.title('Metrics over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.grid(True)  

def plot_confusion_matrix_binary(confusion_matrix):
    """
    Plot the confusion matrix for classification models.
    """
    plt.figure(figsize=(3, 3))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusion_matrix(confusion_matrix):
    """
    Plot the confusion matrix for classification models.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def visualize_computation_graph(tensor, highlight_start=False):
    """
    Visualize the computation graph, starting from a given Tensor node.

    Args:
        tensor (Tensor): The starting Tensor node for the computation graph, typically the loss node.
        highlight_start (bool, optional): Whether to highlight the starting tensor and its expansion path.
    """
    # Highlight the starting tensor if specified
    highlight_tensor = tensor if highlight_start else None
    graph = nx.DiGraph()
    visited = set()
    
    # Use DFS to traverse the entire computation graph and build nodes and edges
    def add_nodes_edges(t):
        if id(t) in visited:
            return
        visited.add(id(t))
        
        # Add current tensor node information
        label = "Tensor\nShape: {}".format(t.shape)
        if t.requires_grad:
            label += "\nGrad: Required"
        else:
            label += "\nGrad: Not Required"
        graph.add_node(id(t), label=label, highlighted=np.array_equal(t, highlight_tensor))
        
        # Traverse parent nodes and add edges and parent node information
        if t.op is not None:
            for parent in t.parents:
                graph.add_edge(id(parent), id(t), label=t.op.__name__ if t.op else "")
                add_nodes_edges(parent)
    
    # Start recursive traversal from the given tensor
    add_nodes_edges(tensor)
    
    # Plot the computation graph
    plt.figure(figsize=(8, 6))
    
    # Use spring_layout to ensure nodes are as non-overlapping as possible
    pos = nx.spring_layout(graph, k=0.4, iterations=50, seed=0)
    
    # Get labels for nodes and edges
    labels = nx.get_node_attributes(graph, 'label')
    edge_labels = nx.get_edge_attributes(graph, 'label')
    
    # Determine node colors; highlight specific nodes if required
    node_colors = ["lightgreen" if graph.nodes[node].get('highlighted', False) else "skyblue" for node in graph.nodes()]
    
    # Draw nodes and edges
    nx.draw(graph, pos, with_labels=True, labels=labels, node_size=3000, node_color=node_colors, font_size=6, edgecolors='k')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6, font_color='red')
    
    plt.title("Computation Graph")
    plt.axis('off')


# Example usage (in train.ipynb):
# from visualization import plot_samples, plot_loss_curve, plot_regression_line, plot_loss_surface_with_path
# plot_samples(X_train, y_train)
# plot_loss_curve(Loss)
# plot_regression_line(X_train, y_train, w1_pred, w0_pred)
# plot_loss_surface_with_path(model, X_train, y_train, criterion, param_history)