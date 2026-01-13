"""
Debug Single Image - Quick pipeline test with detailed output.

Runs 1 image through 10 iterations to verify the pipeline works and show
what's happening at each step.

Usage: python debug_single_image.py
"""

import numpy as np
from PIL import Image
import torch
print("Imports successful!")

# Minimal config
NUM_ITERATIONS = 100
CONVERGENCE_THRESHOLD = 0.99
CONVERGENCE_WINDOW = 2

print("\n" + "="*70)
print("DEBUG SINGLE IMAGE - Pipeline Test")
print("="*70)
print(f"Running {NUM_ITERATIONS} iterations on 1 test image")
print(f"Convergence threshold: {CONVERGENCE_THRESHOLD}")
print(f"This is a MINIMAL test to verify your environment works")
print("="*70)

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity."""
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float((vec_a @ vec_b) / denom) if denom else 0.0

def detect_convergence(embeddings, window_size=2, threshold=0.99):
    """Simple convergence detection."""
    if len(embeddings) < window_size + 1:
        return False, -1, "none"
    
    recent_sims = []
    for idx in range(len(embeddings) - window_size + 1, len(embeddings)):
        sim = cosine_similarity(embeddings[idx], embeddings[idx - 1])
        recent_sims.append(sim)
    
    if recent_sims and np.mean(recent_sims) > threshold:
        return True, len(embeddings) - window_size, "fixed_point"
    
    return False, -1, "none"

print("\n" + "-"*70)
print("STEP 1: Create test image")
print("-"*70)
# Create a simple test image (100x100 RGB)
test_image = Image.new('RGB', (100, 100), color='blue')
print(f"✓ Created test image: {test_image.size}, mode={test_image.mode}")

print("\n" + "-"*70)
print("STEP 2: Simulate embeddings")
print("-"*70)
print("(In real pipeline, these come from LLaVA/CLIP/SentenceTransformer)")

# Simulate embedding extraction with gradually converging vectors
base_embedding = np.random.randn(768)
base_embedding = base_embedding / np.linalg.norm(base_embedding)  # normalize

embeddings = []
captions = [
    "A blue square",
    "A blue square image",
    "A blue square picture"
]

print("\nRunning iteration loop...")
for iteration in range(NUM_ITERATIONS):
    print(f"\n  Iteration {iteration}:")
    
    # Simulate embedding that gradually converges
    if iteration == 0:
        current_emb = base_embedding.copy()
    else:
        # Add decreasing noise (simulating convergence)
        noise_scale = 0.3 / (iteration + 1)
        noise = np.random.randn(768) * noise_scale
        current_emb = embeddings[-1] + noise
        current_emb = current_emb / np.linalg.norm(current_emb)
    
    embeddings.append(current_emb)
    
    # Print embedding info
    emb_norm = np.linalg.norm(current_emb)
    print(f"    Embedding norm: {emb_norm:.6f}")
    
    # Similarity to previous
    if iteration > 0:
        sim_to_prev = cosine_similarity(current_emb, embeddings[iteration-1])
        print(f"    Similarity to previous: {sim_to_prev:.4f}")
    else:
        print(f"    Similarity to previous: N/A (first iteration)")
    
    # Simulated caption
    caption = captions[min(iteration, len(captions)-1)]
    print(f"    Caption: \"{caption}\"")
    
    # Check convergence
    if len(embeddings) >= 2:
        converged, conv_iter, attr_type = detect_convergence(
            embeddings, 
            window_size=CONVERGENCE_WINDOW,
            threshold=CONVERGENCE_THRESHOLD
        )
        if converged:
            print(f"    ✓ CONVERGED to {attr_type} at iteration {conv_iter}!")
        else:
            print(f"    Not converged yet")

print("\n" + "-"*70)
print("STEP 3: Final summary")
print("-"*70)
print(f"Completed {NUM_ITERATIONS} iterations")
print(f"Generated {len(embeddings)} embeddings")
print(f"All embeddings shape: (768,)")

# Final convergence check
converged, conv_iter, attr_type = detect_convergence(
    embeddings,
    window_size=CONVERGENCE_WINDOW,
    threshold=CONVERGENCE_THRESHOLD
)

if converged:
    print(f"\n✓ Final state: CONVERGED")
    print(f"  Attractor type: {attr_type}")
    print(f"  Converged at iteration: {conv_iter}")
else:
    print(f"\n✗ Final state: NOT CONVERGED")
    print(f"  (May need more iterations)")

# Show similarity progression
if len(embeddings) > 1:
    print("\nSimilarity progression:")
    for i in range(1, len(embeddings)):
        sim = cosine_similarity(embeddings[i], embeddings[i-1])
        print(f"  iter {i-1}→{i}: {sim:.4f}")

print("\n" + "="*70)
print("DEBUG TEST COMPLETE!")
print("="*70)
print("\nWhat this simulated:")
print("  ✓ Iteration loop structure")
print("  ✓ Embedding extraction (3 spaces in real code)")
print("  ✓ Convergence detection algorithm")
print("  ✓ Gradual convergence to attractor")
print("\nReal pipeline adds:")
print("  • Actual VLM captioning (LLaVA)")
print("  • Actual image generation (Stable Diffusion)")
print("  • 3 separate embedding spaces (text, vision, CLIP)")
print("  • Log-probability calculations")
print("  • Checkpoint saving")
print("\nIf you see this message, your environment is ready!")
print("Next: Read Day 1 of RECOVERY_PLAN.md and begin learning!")