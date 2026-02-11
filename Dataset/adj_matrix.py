import numpy as np
import h5py

# Define the joints with new indexing
joints = [
    "Center", "ShoulderLeft", "ElbowLeft", "WristLeft",
    "ShoulderRight", "ElbowRight", "WristRight"
]

# Initialize a 7x7 adjacency matrix with zeros
adjacency_matrix = np.zeros((7, 7))

# Define the new connections (with Center only connected to the shoulders)
connections = [
    (0, 1),  # Center ↔ ShoulderLeft
    (1, 2),  # ShoulderLeft ↔ ElbowLeft
    (2, 3),  # ElbowLeft ↔ WristLeft
    (0, 4),  # Center ↔ ShoulderRight
    (4, 5),  # ShoulderRight ↔ ElbowRight
    (5, 6)   # ElbowRight ↔ WristRight
]

# Set the connections in the adjacency matrix (undirected graph)
for parent, child in connections:
    adjacency_matrix[parent][child] = 1
    adjacency_matrix[child][parent] = 1  # undirected graph

# Now we will save this adjacency matrix to an HDF5 file
output_file = "HDF5_Dataset/adjacency_matrix.h5"

# Create an HDF5 file and store the adjacency matrix
with h5py.File(output_file, 'w') as f:
    # Save the adjacency matrix
    f.create_dataset("adjacency_matrix", data=adjacency_matrix)

# Verify by opening the file and printing the adjacency matrix
with h5py.File(output_file, 'r') as f:
    adj_matrix = f["adjacency_matrix"][:]
    print("Adjacency Matrix saved in HDF5 file:")
    print(adj_matrix)
