import numpy as np
from scipy.spatial.distance import cosine
import time
from tqdm import tqdm

def generate_large_matrix(rows, cols):
    return np.random.rand(rows, cols)

def cosine_similarity_chunk(matrix, chunk_size=10000):
    n = matrix.shape[0]
    similarities = np.zeros((n, n))
    
    for i in tqdm(range(0, n, chunk_size)):
        chunk = matrix[i:i+chunk_size]
        for j in range(0, n, chunk_size):
            if i > j:
                continue
            chunk2 = matrix[j:j+chunk_size]
            sim = 1 - np.array([[cosine(a, b) for b in chunk2] for a in chunk])
            similarities[i:i+chunk_size, j:j+chunk_size] = sim
            if i != j:
                similarities[j:j+chunk_size, i:i+chunk_size] = sim.T
    
    return similarities

def main():
    rows, cols = 1_000_000, 300
    chunk_size = 10000  # Adjust this based on your available memory
    
    print(f"Generating matrix of shape ({rows}, {cols})...")
    start_time = time.time()
    matrix = generate_large_matrix(rows, cols)
    print(f"Matrix generation time: {time.time() - start_time:.2f} seconds")

    print("Calculating cosine similarities...")
    start_time = time.time()
    similarities = cosine_similarity_chunk(matrix, chunk_size)
    print(f"Similarity calculation time: {time.time() - start_time:.2f} seconds")

    print(f"Shape of similarity matrix: {similarities.shape}")
    print("Sample of similarity matrix (top-left 5x5):")
    print(similarities[:5, :5])

if __name__ == "__main__":
    main()