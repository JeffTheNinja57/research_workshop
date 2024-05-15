import pickle
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the stimuli embeddings
with open("~/one_dim_embeddings5000.pkl", 'rb') as file:
    image_embeddings = pickle.load(file)

# Load the fMRI embeddings
with open('~/fMRI_one_dim_embeddings5000.pkl', 'rb') as file:
    fMRI_embeddings = pickle.load(file)

def compute_rdm(data):
    """
    Compute the Representational Dissimilarity Matrix (RDM) from the embeddings.

    Parameters:
    data (numpy.ndarray): A 2D array where each row is an embedding.

    Returns:
    numpy.ndarray: The RDM matrix.
    """
    # Compute the correlation matrix
    correlation_matrix = np.corrcoef(data)
    
    # Convert correlation to dissimilarity
    dissimilarity_matrix = 1 - correlation_matrix
    
    # Ensure no negative values
    dissimilarity_matrix[dissimilarity_matrix < 0] = 0
    
    return dissimilarity_matrix

def representational_similarity_analysis(matrix1, matrix2):
    """
    Calculate the representational similarity between two correlation matrices
    using the Mantel test (Pearson correlation of upper triangles of the matrices).
    
    Args:
    matrix1 (numpy.ndarray): First square correlation matrix.
    matrix2 (numpy.ndarray): Second square correlation matrix.
    
    Returns:
    float: Pearson correlation coefficient.
    float: P-value of the correlation.
    """
    # Extract the upper triangle of the matrices, excluding the diagonal
    upper_tri1 = squareform(matrix1, checks=False)
    upper_tri2 = squareform(matrix2, checks=False)
    
    # Compute the Pearson correlation between these upper triangles
    correlation, p_value = pearsonr(upper_tri1, upper_tri2)
    return correlation, p_value

def plot_matrices(matrix1, matrix2, title1="Image Data", title2="fMRI Data", correlation = float, p_value = float):
    """
    Plot two correlation matrices.
    
    Args:
    matrix1 (numpy.ndarray): First correlation matrix.
    matrix2 (numpy.ndarray): Second correlation matrix.
    title1 (str): Title for the first matrix plot.
    title2 (str): Title for the second matrix plot.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    im1 = axs[0].imshow(matrix1, cmap='viridis', vmax=1, vmin=0)
    axs[0].set_title(title1)
    fig.colorbar(im1, ax=axs[0])
    
    im2 = axs[1].imshow(matrix2, cmap='viridis',vmax=1, vmin=0)
    axs[1].set_title(title2)
    fig.colorbar(im2, ax=axs[1])
    
    plt.suptitle("Representational Dissimilarity Matrices", fontsize=16)

    bottom_text = f'A dissimilarity value of: r = {correlation:.3e}, and p value of: p = {p_value:.3e}'
    plt.figtext(0.5, 0.01, bottom_text, wrap=True, horizontalalignment='center', fontsize=12)

    plt.savefig("correlation_matrices.png")
    plt.show()

fMRI_tensors = []

x = 0
for name in fMRI_embeddings.keys():
    if x == 4916:
        break
    fMRI_tensors.append(torch.tensor(fMRI_embeddings[name]).to(device))
    x += 1

combined_embeddings = torch.stack(fMRI_tensors)

image_tensors = []

i = 0
for name in image_embeddings.keys():
    if i == 4916:
        break
    image_tensors.append(torch.tensor(image_embeddings[name]).to(device))
    i += 1

img_combined_embeddings = torch.stack(image_tensors)

# Move combined embeddings to CPU for NumPy operations
img_combined_embeddings_cpu = img_combined_embeddings.cpu().numpy()
combined_embeddings_cpu = combined_embeddings.cpu().numpy()

print(img_combined_embeddings_cpu.shape, combined_embeddings_cpu.shape)

# Compute correlation matrices
correlation_matrix1 = compute_rdm(img_combined_embeddings_cpu)
correlation_matrix2 = compute_rdm(combined_embeddings_cpu)

print(correlation_matrix1.shape, correlation_matrix2.shape)

# Perform RSA
correlation, p_value = representational_similarity_analysis(correlation_matrix1, correlation_matrix2)
print(f"RSA Correlation: {correlation}, P-value: {p_value}")

# Plot the matrices for visualization
plot_matrices(correlation_matrix1, correlation_matrix2, correlation = correlation, p_value = p_value)
