"""
Trace Embeddings - Follow one image's complete journey

Extracts and visualizes one specific image's trajectory through all iterations,
showing all three embedding spaces and convergence detection.

Usage: 
    python trace_embeddings.py --image-id 42 --run-dir /path/to/run
    python trace_embeddings.py --image-id 1  # Uses default run dir
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_checkpoints(run_dir):
    """Load all checkpoint files and merge, preserving per-iteration image ids."""
    checkpoints = sorted(run_dir.glob("checkpoint_imgs*.npz"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    
    embeddings_by_iter = {}
    image_ids_by_iter = {}
    
    for ckpt in checkpoints:
        data = np.load(ckpt, allow_pickle=True)
        embs = data["embeddings_by_iter"].item()
        ids_by_iter = data.get("image_ids_by_iter")
        ids_map = ids_by_iter.item() if ids_by_iter is not None else None
        fallback_ids = data["image_ids"]
        for iter_idx, arr in embs.items():
            iter_idx_int = int(iter_idx)
            embeddings_by_iter.setdefault(iter_idx_int, []).append(np.array(arr))
            if ids_map is not None:
                ids_for_iter = ids_map.get(iter_idx) or ids_map.get(str(iter_idx))
                if ids_for_iter is not None:
                    image_ids_by_iter.setdefault(iter_idx_int, []).append(np.array(ids_for_iter, dtype=np.int32))
                    continue
            # Fallback: assume shared ids list
            else:
                image_ids_by_iter.setdefault(iter_idx_int, []).append(np.array(fallback_ids, dtype=np.int32))
    
    # Stack all checkpoint parts
    for iter_idx in embeddings_by_iter:
        embeddings_by_iter[iter_idx] = np.vstack(embeddings_by_iter[iter_idx])
        if iter_idx in image_ids_by_iter:
            image_ids_by_iter[iter_idx] = np.concatenate(image_ids_by_iter[iter_idx])
    
    return embeddings_by_iter, image_ids_by_iter

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity."""
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float((vec_a @ vec_b) / denom) if denom else 0.0

def detect_convergence_point(embeddings, window=3, threshold=0.95):
    """Find when/if convergence occurred."""
    if len(embeddings) < window + 1:
        return None, "none"
    
    for idx in range(window, len(embeddings)):
        recent_sims = []
        for j in range(idx - window + 1, idx + 1):
            sim = cosine_similarity(embeddings[j], embeddings[j-1])
            recent_sims.append(sim)
        
        if np.mean(recent_sims) > threshold:
            return idx - window, "fixed_point"
    
    return None, "none"

def main():
    parser = argparse.ArgumentParser(description="Trace one image through all iterations")
    parser.add_argument("--image-id", type=int, required=True, help="Image ID to trace")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory")
    args = parser.parse_args()
    
    # Default run dir if not provided
    if args.run_dir is None:
        default = Path("experiments/10_iter_12.17/20251217_183826_iter10_img500")
        if default.exists():
            run_dir = default
            print(f"Using default run directory: {run_dir}")
        else:
            print("Error: No run directory specified and default not found")
            print("Please specify --run-dir /path/to/run")
            return
    else:
        run_dir = Path(args.run_dir)
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return
    
    print("="*70)
    print(f"TRACE EMBEDDINGS - Image {args.image_id}")
    print("="*70)
    
    # Load CSV for metrics
    csv_files = list(run_dir.glob("*_iteration_results.csv"))
    if not csv_files:
        print("Error: No iteration results CSV found")
        return
    
    csv_path = csv_files[0]
    print(f"\nLoading CSV: {csv_path.name}")
    df = pd.read_csv(csv_path)
    
    # Filter for this image
    image_df = df[df["image_id"] == args.image_id].sort_values("iteration")
    
    if len(image_df) == 0:
        print(f"Error: Image ID {args.image_id} not found in results")
        print(f"Available image IDs: {sorted(df['image_id'].unique())[:10]}...")
        return
    
    print(f"Found {len(image_df)} iterations for image {args.image_id}")
    
    # Load CLIP embeddings
    print("\nLoading CLIP embeddings...")
    embeddings_by_iter, image_ids_by_iter = load_checkpoints(run_dir)
    
    # Collect available ids
    available_ids = set()
    for ids_arr in image_ids_by_iter.values():
        available_ids.update([int(x) for x in ids_arr])
    if args.image_id not in available_ids:
        print(f"Error: Image {args.image_id} not found in checkpoints")
        return
    
    print(f"Image {args.image_id} found in checkpoints")
    
    # Extract this image's CLIP trajectory
    clip_trajectory = []
    for iter_idx in sorted(embeddings_by_iter.keys()):
        ids_arr = image_ids_by_iter.get(iter_idx)
        if ids_arr is None:
            continue
        match = np.where(ids_arr == args.image_id)[0]
        if len(match) == 0:
            print(f"  Warning: missing embedding for image {args.image_id} at iteration {iter_idx}")
            continue
        clip_trajectory.append(embeddings_by_iter[iter_idx][match[0]])
    
    # Print journey summary
    print("\n" + "-"*70)
    print("IMAGE JOURNEY SUMMARY")
    print("-"*70)
    
    for idx, row in image_df.iterrows():
        it = int(row["iteration"])
        caption = row["caption"]
        sim_orig = row.get("similarity_to_original", np.nan)
        sim_prev = row.get("similarity_to_previous", np.nan)
        clip_sim_orig = row.get("clip_sim_to_original", np.nan)
        
        print(f"\nIteration {it}:")
        print(f"  Caption: {caption[:60]}...")
        print(f"  Text sim to original: {sim_orig:.4f}")
        print(f"  CLIP sim to original: {clip_sim_orig:.4f}")
        if it > 0:
            print(f"  Text sim to previous: {sim_prev:.4f}")
        
        # Check convergence in CSV
        if row.get("converged", False):
            print(f"  ✓ CONVERGED ({row.get('attractor_type', 'unknown')}) at iter {row.get('convergence_iteration', -1)}")
    
    # Analyze CLIP trajectory
    print("\n" + "-"*70)
    print("CLIP EMBEDDING ANALYSIS")
    print("-"*70)
    
    clip_norms = [np.linalg.norm(emb) for emb in clip_trajectory]
    print(f"Embedding norms: {[f'{n:.4f}' for n in clip_norms]}")
    
    # Step sizes
    step_sizes = []
    for i in range(1, len(clip_trajectory)):
        dist = 1.0 - cosine_similarity(clip_trajectory[i], clip_trajectory[i-1])
        step_sizes.append(dist)
    
    print("\nStep sizes (cosine distance):")
    for i, step in enumerate(step_sizes):
        print(f"  iter {i}→{i+1}: {step:.4f}")
    
    # Detect convergence
    conv_iter, conv_type = detect_convergence_point(clip_trajectory)
    if conv_iter is not None:
        print(f"\nCLIP convergence detected: {conv_type} at iteration {conv_iter}")
    else:
        print("\nNo CLIP convergence detected in this trajectory")
    
    # Visualize
    print("\n" + "-"*70)
    print("CREATING VISUALIZATION")
    print("-"*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Similarity to original
    axes[0, 0].plot(image_df["iteration"], image_df["similarity_to_original"], 
                    'o-', label="Text (caption)")
    if "clip_sim_to_original" in image_df.columns:
        axes[0, 0].plot(image_df["iteration"], image_df["clip_sim_to_original"],
                        's-', label="CLIP")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Similarity to original")
    axes[0, 0].set_title(f"Drift from Original (Image {args.image_id})")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Step sizes
    axes[0, 1].plot(range(1, len(step_sizes)+1), step_sizes, 'o-', color='orange')
    axes[0, 1].set_xlabel("Iteration transition")
    axes[0, 1].set_ylabel("Cosine distance")
    axes[0, 1].set_title("Step Size (CLIP)")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Embedding norm
    axes[1, 0].plot(range(len(clip_norms)), clip_norms, 'o-', color='green')
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("L2 Norm")
    axes[1, 0].set_title("CLIP Embedding Norm")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: BERTScore
    if "bert_f1" in image_df.columns:
        axes[1, 1].plot(image_df["iteration"], image_df["bert_f1"], 'o-', color='purple')
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("BERTScore F1")
        axes[1, 1].set_title("Caption Semantic Similarity")
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = run_dir / "plots" / f"trace_image_{args.image_id}.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved visualization: {output_path}")
    
    print("\n" + "="*70)
    print("TRACE COMPLETE")
    print("="*70)
    print(f"\nThis image:")
    print(f"  • Started with caption: {image_df.iloc[0]['caption'][:50]}...")
    print(f"  • Ended with caption: {image_df.iloc[-1]['caption'][:50]}...")
    print(f"  • Final text similarity: {image_df.iloc[-1]['similarity_to_original']:.4f}")
    print(f"  • Final CLIP similarity: {image_df.iloc[-1].get('clip_sim_to_original', 'N/A')}")
    print(f"  • Largest step: iter {np.argmax(step_sizes)}→{np.argmax(step_sizes)+1} ({max(step_sizes):.4f})")
    
    if conv_iter is not None:
        print(f"  • Converged at iteration {conv_iter}")

if __name__ == "__main__":
    main()
