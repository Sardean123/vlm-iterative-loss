import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score


def load_all_checkpoints(run_dir: Path) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray]:
    """
    Merge all checkpoint NPZs into per-iteration embedding arrays, preserving per-iter image ids when available.
    """
    checkpoint_files = sorted(run_dir.glob("checkpoint_imgs*.npz"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint NPZs found in {run_dir}")

    embeddings_by_iter: Dict[int, List[np.ndarray]] = {}
    image_ids_by_iter: Dict[int, List[np.ndarray]] = {}
    all_image_ids: List[int] = []

    for ckpt in checkpoint_files:
        data = np.load(ckpt, allow_pickle=True)
        ckpt_embs = data["embeddings_by_iter"].item()
        ids_by_iter = data.get("image_ids_by_iter")
        ids_map = ids_by_iter.item() if ids_by_iter is not None else None
        ckpt_image_ids = data["image_ids"]

        for iter_idx, embs in ckpt_embs.items():
            iter_idx_int = int(iter_idx)
            embeddings_by_iter.setdefault(iter_idx_int, []).append(np.array(embs))

            if ids_map is not None:
                ids_for_iter = None
                if iter_idx in ids_map:
                    ids_for_iter = ids_map[iter_idx]
                elif str(iter_idx) in ids_map:
                    ids_for_iter = ids_map[str(iter_idx)]
                if ids_for_iter is not None:
                    image_ids_by_iter.setdefault(iter_idx_int, []).append(np.array(ids_for_iter, dtype=np.int32))
                    continue
            # Fallback: assume shared ids list
            image_ids_by_iter.setdefault(iter_idx_int, []).append(np.array(ckpt_image_ids, dtype=np.int32))

        all_image_ids.extend([int(x) for x in ckpt_image_ids])

    merged_embeddings: Dict[int, np.ndarray] = {}
    merged_ids: Dict[int, np.ndarray] = {}
    for iter_idx, parts in embeddings_by_iter.items():
        merged_embeddings[iter_idx] = np.vstack(parts)
        if iter_idx in image_ids_by_iter:
            merged_ids[iter_idx] = np.concatenate(image_ids_by_iter[iter_idx])

    return merged_embeddings, merged_ids, np.array(all_image_ids, dtype=np.int32)


def resolve_embeddings_dir(run_dir: Path) -> Path:
    """
    Resolve where CLIP embedding checkpoints live.
    If run_dir/CLIP_embeddings exists, use that subfolder; otherwise use run_dir directly.
    """
    clip_dir = run_dir / "CLIP_embeddings"
    if clip_dir.exists():
        return clip_dir
    return run_dir


def reduce_embeddings(
    embeddings: np.ndarray,
    n_components: int,
    pca_model: Optional[PCA] = None,
    apply_pca: bool = True,
) -> Tuple[np.ndarray, Optional[PCA], float]:
    """
    Reduce embeddings with PCA (optionally using a prefit model), or return raw embeddings when PCA is disabled.
    Returns reduced embeddings, fitted/used PCA, and explained variance.
    """
    if not apply_pca or n_components <= 0:
        return embeddings, None, 1.0  # type: ignore

    if pca_model is not None:
        reduced = pca_model.transform(embeddings)
        explained = float(pca_model.explained_variance_ratio_.sum())
        return reduced, pca_model, explained

    if embeddings.shape[1] <= n_components:
        return embeddings, None, 1.0  # type: ignore
    pca = PCA(n_components=n_components, random_state=420, whiten=False)
    reduced = pca.fit_transform(embeddings)
    explained = float(pca.explained_variance_ratio_.sum())
    if explained < 0.85:
        print(f"  ⚠️  Warning: PCA only explains {explained:.1%} of variance")
    return reduced, pca, explained


def fit_global_pca(embeddings_by_iter: Dict[int, np.ndarray], n_components: int) -> Tuple[PCA, float]:
    """
    Fit a PCA model across all iterations to ensure a consistent space.
    """
    all_emb = np.vstack([emb for emb in embeddings_by_iter.values()])
    pca = PCA(n_components=n_components, random_state=420, whiten=False)
    pca.fit(all_emb)
    explained = float(pca.explained_variance_ratio_.sum())
    if explained < 0.85:
        print(f"  ⚠️  Warning: global PCA only explains {explained:.1%} of variance")
    return pca, explained


def align_clusters_across_iterations(
    labels_by_iter: Dict[int, np.ndarray],
    k: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
    """
    Align cluster labels over iterations using the Hungarian algorithm based on overlap.

    Returns aligned labels and the mapping from reference labels to original labels
    for each iteration.
    """
    iters = sorted(labels_by_iter.keys())
    if not iters:
        return {}, {}

    aligned: Dict[int, np.ndarray] = {}
    permutations: Dict[int, Dict[int, int]] = {}

    first_iter = iters[0]
    aligned[first_iter] = labels_by_iter[first_iter].copy()
    permutations[first_iter] = {i: i for i in range(k)}

    for prev_it, curr_it in zip(iters, iters[1:]):
        prev_labels = aligned[prev_it]
        curr_labels = labels_by_iter[curr_it]
        cost = np.zeros((k, k))
        for c_prev in range(k):
            for c_curr in range(k):
                overlap = np.sum((prev_labels == c_prev) & (curr_labels == c_curr))
                cost[c_prev, c_curr] = -overlap
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {int(prev): int(curr) for prev, curr in zip(row_ind, col_ind)}
        relabeled = np.full_like(curr_labels, fill_value=-1)
        for target_label, source_label in mapping.items():
            relabeled[curr_labels == source_label] = target_label
        aligned[curr_it] = relabeled
        permutations[curr_it] = mapping
    return aligned, permutations


def reorder_centroids(
    centroids: np.ndarray,
    assignment: Dict[int, int],
    k: int,
) -> np.ndarray:
    """
    Reorder centroids to match the aligned label ordering.
    """
    if centroids.shape[0] != k:
        return centroids
    reordered = np.array(centroids, copy=True)
    for target in range(k):
        source = assignment.get(target, target)
        if source < centroids.shape[0]:
            reordered[target] = centroids[source]
    return reordered


def analyze_iteration(
    embeddings: np.ndarray,
    k_values: List[int],
    pca_components: int = 256,
    pca_model: Optional[PCA] = None,
    force_k: Optional[int] = None,
    apply_pca: bool = True,
) -> Dict:
    """
    Run optional PCA + K-Means + clustering metrics for a single iteration snapshot.
    """
    n_samples, n_dim = embeddings.shape
    if n_samples < 2:
        return {
            "n_samples": n_samples,
            "embedding_dim": n_dim,
            "pairwise_mean": None,
            "pairwise_std": None,
            "pca_explained_variance": None,
            "optimal_k": None,
            "best_silhouette": None,
            "mean_distance_to_centroid": None,
            "cluster_sizes": {},
            "k_results": [],
            "centroids": None,
            "labels": None,
        }

    # Pairwise dispersion (euclidean distance on original normalized embeddings)
    pairwise = pdist(embeddings, metric="euclidean")
    pairwise_mean = float(np.mean(pairwise))
    pairwise_std = float(np.std(pairwise))

    # PCA reduction (use shared PCA if provided)
    emb_reduced, pca, explained_var = reduce_embeddings(
        embeddings,
        pca_components,
        pca_model=pca_model,
        apply_pca=apply_pca,
    )
    used_pca_components = emb_reduced.shape[1]
    if not apply_pca:
        explained_var = None

    best_k = None
    best_silhouette = -1.0
    best_labels = None
    best_centroids = None
    k_results = []

    k_candidates = [force_k] if force_k is not None else k_values
    for k in k_candidates:
        if k is None:
            continue
        if k >= n_samples:
            continue
        kmeans = KMeans(
            n_clusters=k,
            n_init=20,
            max_iter=500,
            random_state=420,
        )
        labels = kmeans.fit_predict(emb_reduced)

        silhouette = silhouette_score(emb_reduced, labels, metric="euclidean")
        davies_bouldin = davies_bouldin_score(emb_reduced, labels)
        inertia = float(kmeans.inertia_)

        k_results.append(
            {
                "k": int(k),
                "silhouette": float(silhouette),
                "davies_bouldin": float(davies_bouldin),
                "inertia": inertia,
            }
        )

        if silhouette > best_silhouette:
            best_silhouette = float(silhouette)
            best_k = int(k)
            best_labels = labels
            best_centroids = kmeans.cluster_centers_

    # Compute tightness for the best clustering (in PCA space)
    mean_distance_to_centroid = None
    cluster_sizes = {}
    if best_labels is not None and best_centroids is not None:
        dists = []
        unique, counts = np.unique(best_labels, return_counts=True)
        cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts)}
        for cluster_id in unique:
            mask = best_labels == cluster_id
            if not np.any(mask):
                continue
            distances = np.linalg.norm(emb_reduced[mask] - best_centroids[int(cluster_id)], axis=1)
            dists.extend(distances.tolist())
        if dists:
            mean_distance_to_centroid = float(np.mean(dists))

    return {
        "n_samples": n_samples,
        "embedding_dim": n_dim,
        "pairwise_mean": pairwise_mean,
        "pairwise_std": pairwise_std,
        "pca_components": used_pca_components,
        "pca_explained_variance": explained_var,
        "optimal_k": best_k,
        "best_silhouette": best_silhouette,
        "mean_distance_to_centroid": mean_distance_to_centroid,
        "cluster_sizes": cluster_sizes,
        "k_results": k_results,
        "centroids": best_centroids.tolist() if best_centroids is not None else None,
        "labels": best_labels.tolist() if best_labels is not None else None,
    }


def run_clustering_analysis(
    embeddings_by_iter: Dict[int, np.ndarray],
    k_values: List[int],
    pca_components: int,
    pca_scope: str = "global",
    align_k: Optional[int] = None,
) -> Tuple[Dict[int, Dict], Optional[PCA]]:
    """
    Run clustering analysis across all iterations.
    """
    results: Dict[int, Dict] = {}
    shared_pca: Optional[PCA] = None

    if pca_scope == "global":
        shared_pca, global_explained = fit_global_pca(embeddings_by_iter, pca_components)
        print(f"Using global PCA ({shared_pca.n_components_} comps), explained variance: {global_explained:.2%}")
    elif pca_scope == "none":
        print("Skipping PCA; running K-Means on raw embeddings.")
    elif pca_scope != "per-iteration":
        raise ValueError(f"Unsupported pca_scope: {pca_scope}, try `global`, `per-iteration`, or `none`")

    apply_pca = pca_scope != "none"

    for iter_idx in sorted(embeddings_by_iter.keys()):
        emb = embeddings_by_iter[iter_idx]
        print(f"\nAnalyzing iteration {iter_idx} with {len(emb)} samples...")
        iter_result = analyze_iteration(
            emb,
            k_values,
            pca_components=pca_components,
            pca_model=shared_pca if apply_pca else None,
            force_k=align_k,
            apply_pca=apply_pca,
        )
        results[iter_idx] = iter_result

        pw_mean = iter_result["pairwise_mean"]
        best_k = iter_result["optimal_k"]
        best_sil = iter_result["best_silhouette"]
        explained = iter_result.get("pca_explained_variance")
        if pw_mean is not None:
            print(f"  Mean pairwise distance: {pw_mean:.4f}")
        if explained is not None:
            print(f"  PCA explained variance (@{iter_result.get('pca_components')} comps): {explained:.2%}")
        print(f"  Optimal k: {best_k}")
        if best_sil is not None:
            print(f"  Best silhouette: {best_sil:.4f}")

    # Align clusters across iterations if requested
    if align_k is not None:
        labels_for_alignment = {
            iter_idx: np.array(r["labels"])
            for iter_idx, r in results.items()
            if r.get("labels") is not None
        }
        if len(labels_for_alignment) == len(results):
            aligned_labels, permutations = align_clusters_across_iterations(labels_for_alignment, align_k)
            for iter_idx, aligned in aligned_labels.items():
                res = results[iter_idx]
                res["labels"] = aligned.tolist()
                centroids = res.get("centroids")
                if centroids is not None:
                    reordered = reorder_centroids(np.array(centroids), permutations.get(iter_idx, {}), align_k)
                    res["centroids"] = reordered.tolist()
            print(f"\nAligned cluster labels across iterations using k={align_k}.")
        else:
            missing = sorted(set(results.keys()) - set(labels_for_alignment.keys()))
            print(f"\n⚠️  Skipping alignment: missing labels for iterations {missing}")

    return results, shared_pca


def plot_convergence_metrics(results_by_iter: Dict[int, Dict], output_dir: Path) -> Path:
    """
    Plot convergence metrics across iterations.
    """
    iterations = sorted(results_by_iter.keys())
    metrics = {
        "pairwise_mean": [],
        "optimal_k": [],
        "best_silhouette": [],
        "mean_distance_to_centroid": [],
    }

    for iter_idx in iterations:
        r = results_by_iter[iter_idx]
        for key in metrics:
            metrics[key].append(r.get(key) if r.get(key) is not None else np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(iterations, metrics["pairwise_mean"], "o-", linewidth=2)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Mean Pairwise Euclidean Distance")
    axes[0, 0].set_title("Image Diversity Over Time")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iterations, metrics["optimal_k"], "o-", linewidth=2, color="orange")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Optimal Number of Clusters")
    axes[0, 1].set_title("Attractor Count Over Time")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(iterations, metrics["best_silhouette"], "o-", linewidth=2, color="green")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Silhouette Score")
    axes[1, 0].set_title("Cluster Quality")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(iterations, metrics["mean_distance_to_centroid"], "o-", linewidth=2, color="red")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Mean Distance to Centroid")
    axes[1, 1].set_title("Cluster Tightness")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "convergence_metrics.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")
    return output_path


def generate_contact_sheets(
    run_dir: Path,
    embeddings_by_iter: Dict[int, np.ndarray],
    image_ids_by_iter: Dict[int, np.ndarray],
    results_by_iter: Dict[int, Dict],
    top_k: int = 5,
    analysis_dir: Optional[Path] = None,
    pca_model: Optional[PCA] = None,
    apply_pca: bool = True,
    iter_limit: Optional[int] = None,
) -> None:
    """
    Generate contact sheets showing nearest images to each centroid.
    """
    images_dir = run_dir / "images"
    base_dir = analysis_dir if analysis_dir is not None else run_dir
    contact_dir = base_dir / "plots" / "attractor_contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    for iter_idx, result in results_by_iter.items():
        if iter_limit is not None and iter_idx >= iter_limit:
            continue
        if result.get("centroids") is None or result.get("labels") is None:
            continue

        centroids = np.array(result["centroids"])
        labels = np.array(result["labels"])
        embeddings = embeddings_by_iter[iter_idx]
        ids_for_iter = image_ids_by_iter.get(iter_idx)
        if ids_for_iter is None or len(ids_for_iter) != len(embeddings):
            print(f"  Warning: image_ids length mismatch at iteration {iter_idx}; skipping contact sheets.")
            continue

        # Recompute embeddings in the same space K-Means used so centroid
        # distances are comparable (PCA reduces dimensionality before KMeans).
        n_components = result.get("pca_components", embeddings.shape[1])
        embeddings_reduced, _, _ = reduce_embeddings(
            embeddings,
            n_components,
            pca_model=pca_model if apply_pca else None,
            apply_pca=apply_pca,
        )

        if embeddings_reduced.shape[1] != centroids.shape[1]:
            print(
                f"  Warning: centroid dimension mismatch at iteration {iter_idx} "
                f"({embeddings_reduced.shape[1]} vs {centroids.shape[1]}). Skipping contact sheets."
            )
            continue

        for cluster_id in range(len(centroids)):
            centroid = centroids[cluster_id]
            distances = np.linalg.norm(embeddings_reduced - centroid, axis=1)
            nearest_indices = np.argsort(distances)[:top_k]

            fig, axes = plt.subplots(1, top_k, figsize=(top_k * 3, 3))
            if top_k == 1:
                axes = [axes]
            fig.suptitle(f"Attractor {cluster_id} - Iteration {iter_idx}")

            for i, idx in enumerate(nearest_indices):
                img_id = ids_for_iter[idx]
                img_files = list(images_dir.glob(f"*_img{img_id:03d}_iter{iter_idx}_*.png"))
                if img_files:
                    try:
                        img = Image.open(img_files[0])
                        axes[i].imshow(img)
                        axes[i].set_title(f"ID {img_id}")
                        axes[i].axis("off")
                    except Exception as e:
                        axes[i].axis("off")
                        print(f"    Warning: could not load image {img_files[0]} ({e})")
                else:
                    axes[i].axis("off")
            plt.tight_layout()
            plt.savefig(contact_dir / f"iter{iter_idx}_cluster{cluster_id}.png", dpi=150)
            plt.close()

    print(f"Contact sheets saved to: {contact_dir}")


def print_summary_statistics(results_by_iter: Dict[int, Dict]) -> None:
    """
    Print a brief convergence summary comparing first and last iterations.
    """
    iterations = sorted(results_by_iter.keys())
    if len(iterations) < 2:
        return

    first_iter = results_by_iter[iterations[0]]
    last_iter = results_by_iter[iterations[-1]]

    print("\n" + "=" * 60)
    print("CONVERGENCE SUMMARY")
    print("=" * 60)

    if first_iter.get("pairwise_mean") is not None and last_iter.get("pairwise_mean") is not None:
        initial_pw = first_iter["pairwise_mean"]
        final_pw = last_iter["pairwise_mean"]
        pw_change = ((final_pw - initial_pw) / initial_pw) * 100 if initial_pw != 0 else 0.0
        print(f"Pairwise distance: {initial_pw:.4f} → {final_pw:.4f} ({pw_change:+.1f}%)")

    if first_iter.get("optimal_k") is not None and last_iter.get("optimal_k") is not None:
        initial_k = first_iter["optimal_k"]
        final_k = last_iter["optimal_k"]
        if initial_k is not None and final_k is not None:
            k_change = final_k - initial_k
            print(f"Optimal clusters: {initial_k} → {final_k} ({k_change:+d})")

    if first_iter.get("best_silhouette") is not None and last_iter.get("best_silhouette") is not None:
        initial_sil = first_iter["best_silhouette"]
        final_sil = last_iter["best_silhouette"]
        if initial_sil is not None and final_sil is not None:
            sil_change = ((final_sil - initial_sil) / abs(initial_sil)) * 100 if initial_sil != 0 else 0.0
            print(f"Silhouette score: {initial_sil:.4f} → {final_sil:.4f} ({sil_change:+.1f}%)")

    print("=" * 60)


def parse_k_values(k_values_str: str) -> List[int]:
    return [int(k.strip()) for k in k_values_str.split(",") if k.strip()]


def load_baseline_npz(npz_path: Path, method: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray]:
    """
    Load baseline embeddings saved by run_baselines.py and reshape into per-iteration dicts.
    Baseline NPZ keys: sd_only, random_caption, fixed_caption.
    """
    data = np.load(npz_path, allow_pickle=True)
    if method not in data:
        raise ValueError(f"Baseline method '{method}' not found in {npz_path}. Available: {list(data.files)}")

    arr = data[method]
    # Baseline NPZ stores object arrays; coerce to a dense float tensor
    if arr.dtype == object:
        arr = np.array(arr.tolist(), dtype=np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)

    if arr.ndim != 3:
        raise ValueError(f"Expected baseline array of shape (num_images, num_iterations, dim); got {arr.shape}")

    num_images, num_iterations, _ = arr.shape
    embeddings_by_iter = {iter_idx: arr[:, iter_idx, :] for iter_idx in range(num_iterations)}
    image_ids_by_iter = {
        iter_idx: np.arange(1, num_images + 1, dtype=np.int32) for iter_idx in range(num_iterations)
    }
    image_ids_flat = np.concatenate(list(image_ids_by_iter.values()))
    return embeddings_by_iter, image_ids_by_iter, image_ids_flat


def main():
    parser = argparse.ArgumentParser(description="Run PCA + K-Means analysis on saved CLIP embeddings.")
    parser.add_argument(
        "run_dir",
        type=Path,
        help=(
            "Experiment root or CLIP_embeddings directory containing checkpoint NPZs. "
            "If CLIP_embeddings exists under run_dir, it will be used."
        ),
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="5,10,15,20,25,30,35,40,45,50",
        help="Comma-separated list of k values to evaluate.",
    )
    parser.add_argument("--pca-components", type=int, default=512, help="Number of PCA components before K-Means.")
    parser.add_argument(
        "--pca-scope",
        type=str,
        choices=["global", "per-iteration", "none"],
        default="global",
        help="Fit PCA across all iterations (global), separately for each iteration, or disable PCA entirely (none).",
    )
    parser.add_argument(
        "--align-k",
        type=int,
        default=None,
        help="Force clustering to this k across iterations and align labels via Hungarian matching.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for JSON results (default: run_dir/analysis/kmeans/<analysis_name>/kmeans_analysis.json).",
    )
    parser.add_argument(
        "--analysis-name",
        type=str,
        default=None,
        help="Name of the analysis subfolder under run_dir/analysis/kmeans (default: timestamp).",
    )
    parser.add_argument("--contact-top-k", type=int, default=5, help="Top-k nearest images per centroid for contact sheets.")
    parser.add_argument(
        "--contact-iter-limit",
        type=int,
        default=None,
        help="Only generate contact sheets for iterations < this value (default: all).",
    )
    parser.add_argument(
        "--skip-contact-sheets",
        action="store_true",
        help="Skip generating contact sheet images.",
    )
    parser.add_argument(
        "--baseline-npz",
        type=Path,
        default=None,
        help="Path to baseline embeddings NPZ (e.g., baseline_clip_embeddings.npz). "
             "When set, load this instead of checkpoint NPZs.",
    )
    parser.add_argument(
        "--baseline-method",
        type=str,
        default="sd_only",
        choices=["sd_only", "random_caption", "fixed_caption"],
        help="Which baseline rollout to analyze inside the baseline NPZ.",
    )

    args = parser.parse_args()
    run_dir: Path = args.run_dir
    k_values = parse_k_values(args.k_values)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    analysis_name = args.analysis_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = run_dir / "analysis" / "kmeans" / analysis_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or (analysis_dir / "kmeans_analysis.json")

    baseline_npz = None
    if args.baseline_npz is not None:
        baseline_npz = args.baseline_npz if args.baseline_npz.is_absolute() else run_dir / args.baseline_npz
        if not baseline_npz.exists():
            raise FileNotFoundError(f"Baseline NPZ not found: {baseline_npz}")
        print(f"Loading baseline embeddings ({args.baseline_method}) from {baseline_npz}")
        embeddings_by_iter, image_ids_by_iter, image_ids_flat = load_baseline_npz(baseline_npz, args.baseline_method)
        embeddings_dir = baseline_npz
    else:
        embeddings_dir = resolve_embeddings_dir(run_dir)
        if embeddings_dir != run_dir:
            print(f"Found CLIP embeddings at {embeddings_dir}")
        print(f"Loading checkpoints from {embeddings_dir} ...")
        embeddings_by_iter, image_ids_by_iter, image_ids_flat = load_all_checkpoints(embeddings_dir)

    print(f"Saving analysis artifacts to {analysis_dir}")
    print(f"Loaded embeddings for iterations: {sorted(embeddings_by_iter.keys())}")
    print(f"Total images covered: {len(np.unique(image_ids_flat))}")

    apply_pca = args.pca_scope != "none"

    results, shared_pca = run_clustering_analysis(
        embeddings_by_iter,
        k_values=k_values,
        pca_components=args.pca_components,
        pca_scope=args.pca_scope,
        align_k=args.align_k,
    )

    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_convergence_metrics(results, plots_dir)
    print_summary_statistics(results)

    # Save centroids/labels for downstream probes
    centroid_dir = analysis_dir / "centroid_labels"
    centroid_dir.mkdir(parents=True, exist_ok=True)
    for iter_idx, result in results.items():
        if result.get("centroids") is None or result.get("labels") is None:
            continue
        np.savez(
            centroid_dir / f"cluster_data_iter{iter_idx}.npz",
            centroids=np.array(result["centroids"]),
            labels=np.array(result["labels"]),
        )

    if baseline_npz is not None and not args.skip_contact_sheets:
        print("Baseline mode detected; skipping contact sheets (baseline runs do not store images).")
        args.skip_contact_sheets = True

    if args.skip_contact_sheets:
        print("Skipping contact sheet generation.")
    else:
        generate_contact_sheets(
            run_dir=run_dir,
            embeddings_by_iter=embeddings_by_iter,
            image_ids_by_iter=image_ids_by_iter,
            results_by_iter=results,
            top_k=args.contact_top_k,
            analysis_dir=analysis_dir,
            pca_model=shared_pca if apply_pca else None,
            apply_pca=apply_pca,
            iter_limit=args.contact_iter_limit,
        )

    output = {
        "run_dir": str(run_dir.resolve()),
        "embeddings_dir": str(embeddings_dir.resolve()),
        "analysis": {"name": analysis_name, "dir": str(analysis_dir.resolve())},
        "k_values": k_values,
        "pca_components": args.pca_components,
        "pca_scope": args.pca_scope,
        "pca_applied": apply_pca,
        "align_k": args.align_k,
        "results_by_iteration": results,
        "baseline": {
            "npz": str(baseline_npz.resolve()),
            "method": args.baseline_method,
        }
        if baseline_npz is not None
        else None,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved analysis results to {output_path}")


if __name__ == "__main__":
    main()
