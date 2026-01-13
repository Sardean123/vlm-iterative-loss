"""
Test Basin Absorption - Verify basin metrics on small dataset

Tests the basin absorption methodology with a subset of images to verify
the math and understanding.

Usage:
    python test_basin_absorption.py --run-dir /path/to/run
    python test_basin_absorption.py  # Uses default run dir
"""

import argparse
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
            else:
                image_ids_by_iter.setdefault(iter_idx_int, []).append(np.array(fallback_ids, dtype=np.int32))
    
    # Stack
    for iter_idx in embeddings_by_iter:
        embeddings_by_iter[iter_idx] = np.vstack(embeddings_by_iter[iter_idx])
        if iter_idx in image_ids_by_iter:
            image_ids_by_iter[iter_idx] = np.concatenate(image_ids_by_iter[iter_idx])
    
    return embeddings_by_iter, image_ids_by_iter

def main():
    parser = argparse.ArgumentParser(description="Test basin absorption metrics")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory")
    parser.add_argument("--num-images", type=int, default=50, help="Number of images to test")
    parser.add_argument("--k", type=int, default=5, help="Number of basins/clusters")
    args = parser.parse_args()
    
    # Default run dir
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
    print("BASIN ABSORPTION TEST")
    print("="*70)
    print(f"Testing on {args.num_images} images")
    print(f"Using k={args.k} basins")
    
    # Load embeddings
    print("\nLoading embeddings...")
    embeddings_by_iter, image_ids_by_iter = load_checkpoints(run_dir)

    # Align to images present in all iterations
    common_ids = None
    for ids_arr in image_ids_by_iter.values():
        ids_set = set(int(x) for x in ids_arr)
        common_ids = ids_set if common_ids is None else common_ids & ids_set
    if not common_ids:
        print("Error: no common image ids across iterations")
        return
    common_ids = sorted(common_ids)

    aligned_embeddings = {}
    for iter_idx, embs in embeddings_by_iter.items():
        ids_arr = image_ids_by_iter.get(iter_idx)
        if ids_arr is None:
            continue
        # Map id -> row index
        id_to_idx = {int(img_id): idx for idx, img_id in enumerate(ids_arr)}
        indices = [id_to_idx[i] for i in common_ids if i in id_to_idx]
        aligned_embeddings[iter_idx] = embs[indices]

    image_ids = np.array(common_ids, dtype=np.int32)

    # Use subset
    n = min(args.num_images, len(image_ids))
    for iter_idx in aligned_embeddings:
        aligned_embeddings[iter_idx] = aligned_embeddings[iter_idx][:n]
    image_ids = image_ids[:n]
    
    print(f"Loaded {n} images across {len(aligned_embeddings)} iterations (aligned on common ids)")
    
    iters = sorted(aligned_embeddings.keys())
    final_iter = max(iters)
    
    print("\n" + "-"*70)
    print("STEP 1: Define Terminal Attractors")
    print("-"*70)
    print(f"Fitting PCA + K-Means on final iteration ({final_iter})...")
    
    # Fit on final iteration
    pca = PCA(n_components=min(128, aligned_embeddings[final_iter].shape[1]), random_state=42)
    Xf = pca.fit_transform(aligned_embeddings[final_iter])
    
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance: {explained:.2%}")
    
    km = KMeans(n_clusters=args.k, n_init=20, max_iter=500, random_state=42)
    labels_final = km.fit_predict(Xf)
    centroids = km.cluster_centers_
    
    print(f"K-Means converged")
    print(f"Final basin sizes: {np.bincount(labels_final)}")
    
    print("\n" + "-"*70)
    print("STEP 2: Assign All Iterations to Terminal Basins")
    print("-"*70)
    
    labels_by_iter = {}
    for iter_idx in iters:
        # Use SAME pca
        Xt = pca.transform(aligned_embeddings[iter_idx])
        
        # Assign to nearest final centroid
        distances = np.linalg.norm(Xt[:, None, :] - centroids[None, :, :], axis=-1)
        labels_by_iter[iter_idx] = distances.argmin(axis=1)
        
        print(f"  Iteration {iter_idx}: assigned to basins")
    
    print("\n" + "-"*70)
    print("STEP 3: Calculate Basin Absorption Metrics")
    print("-"*70)
    
    # Match-final fraction
    print("\nMatch-final fraction (% in terminal basin):")
    match_final = []
    for iter_idx in iters:
        fraction = (labels_by_iter[iter_idx] == labels_final).mean()
        match_final.append(fraction)
        print(f"  Iteration {iter_idx}: {fraction*100:.1f}%")
    
    # Switching rate
    print("\nSwitching rate (% changing basins):")
    for i in range(1, len(iters)):
        prev_iter = iters[i-1]
        curr_iter = iters[i]
        switches = (labels_by_iter[curr_iter] != labels_by_iter[prev_iter]).mean()
        print(f"  Iteration {prev_iter}→{curr_iter}: {switches*100:.1f}%")
    
    # Commitment time
    print("\nCommitment time (when enters terminal basin and stays):")
    commit_times = []
    for img_idx in range(n):
        final_label = labels_final[img_idx]
        commit_time = -1
        
        for t in iters:
            # Check if all future iterations have same label as final
            all_future_match = all(
                labels_by_iter[tt][img_idx] == final_label 
                for tt in iters if tt >= t
            )
            if all_future_match:
                commit_time = t
                break
        
        commit_times.append(commit_time)
    
    commit_times = np.array(commit_times)
    
    # Commitment CDF
    print("\nCommitment CDF:")
    for t in iters:
        fraction = (commit_times <= t).mean()
        print(f"  P(commit_time ≤ {t}): {fraction*100:.1f}%")
    
    # Show some examples
    print("\nExample commitment times:")
    for i in range(min(10, n)):
        img_id = image_ids[i]
        ct = commit_times[i]
        final_basin = labels_final[i]
        print(f"  Image {img_id}: committed to basin {final_basin} at iteration {ct}")
    
    print("\n" + "-"*70)
    print("STEP 4: Within vs Between Basin Distances")
    print("-"*70)
    
    def within_between_distances(embeddings, labels):
        """Calculate within and between basin distances."""
        # Cosine distance matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        sim = normalized @ normalized.T
        dist = 1.0 - sim
        
        # Upper triangle only
        iu = np.triu_indices_from(dist, k=1)
        dist_u = dist[iu]
        
        # Same basin or different?
        same = (labels[iu[0]] == labels[iu[1]])
        
        within = dist_u[same].mean() if np.any(same) else np.nan
        between = dist_u[~same].mean() if np.any(~same) else np.nan
        
        return within, between
    
    print("\nWithin-basin vs between-basin distances:")
    print(f"{'Iter':<6} {'Within':<10} {'Between':<10} {'Ratio':<10}")
    print("-"*40)
    
    for iter_idx in iters:
        within, between = within_between_distances(
            aligned_embeddings[iter_idx],
            labels_final  # Use FINAL basin labels
        )
        ratio = within / between if between > 0 else np.nan
        print(f"{iter_idx:<6} {within:<10.4f} {between:<10.4f} {ratio:<10.4f}")
    
    print("\nInterpretation:")
    print("  • Within decreasing → images converging to basin centers")
    print("  • Between stable/increasing → basins staying separated")
    print("  • Ratio < 1 and decreasing → basin separation improving")
    
    print("\n" + "="*70)
    print("BASIN ABSORPTION TEST COMPLETE")
    print("="*70)
    
    print("\nKey findings:")
    print(f"  • Initial match-final: {match_final[0]*100:.1f}%")
    print(f"  • Final match-final: {match_final[-1]*100:.1f}% (should be 100%)")
    print(f"  • Mean commitment time: {commit_times[commit_times >= 0].mean():.1f}")
    print(f"  • % committed by iter {iters[len(iters)//2]}: {(commit_times <= iters[len(iters)//2]).mean()*100:.1f}%")
    
    print("\nYou now understand:")
    print("  ✓ How terminal attractors are defined (final iter K-Means)")
    print("  ✓ How basin assignment works (nearest centroid in PCA space)")
    print("  ✓ What match-final fraction measures (% in terminal basin)")
    print("  ✓ What switching rate measures (% changing basins)")
    print("  ✓ What commitment time measures (absorption time)")
    print("  ✓ Within vs between basin separation dynamics")

if __name__ == "__main__":
    main()
