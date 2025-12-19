import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score


def load_all_checkpoints(run_dir: Path) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Merge all checkpoint NPZs into per-iteration embedding arrays.
    """
    checkpoint_files = sorted(run_dir.glob("checkpoint_imgs*.npz"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint NPZs found in {run_dir}")

    embeddings_by_iter: Dict[int, List[np.ndarray]] = {}
    all_image_ids: List[int] = []

    for ckpt in checkpoint_files:
        data = np.load(ckpt, allow_pickle=True)
        ckpt_embs = data["embeddings_by_iter"].item()
        ckpt_image_ids = data["image_ids"]

        for iter_idx, embs in ckpt_embs.items():
            embeddings_by_iter.setdefault(int(iter_idx), []).append(np.array(embs))
        all_image_ids.extend([int(x) for x in ckpt_image_ids])

    merged: Dict[int, np.ndarray] = {}
    for iter_idx, parts in embeddings_by_iter.items():
        merged[iter_idx] = np.vstack(parts)

    return merged, np.array(all_image_ids, dtype=np.int32)


def reduce_embeddings(embeddings: np.ndarray, n_components: int) -> Tuple[np.ndarray, Optional[PCA], float]:
    """
    Reduce embeddings with PCA. Returns reduced embeddings, fitted PCA, and explained variance.
    """
    if embeddings.shape[1] <= n_components:
        return embeddings, None, 1.0  # type: ignore
    pca = PCA(n_components=n_components, random_state=420)
    reduced = pca.fit_transform(embeddings)
    explained = float(pca.explained_variance_ratio_.sum())
    if explained < 0.85:
        print(f"  ⚠️  Warning: PCA only explains {explained:.1%} of variance")
    return reduced, pca, explained


def analyze_iteration(
    embeddings: np.ndarray,
    k_values: List[int],
    pca_components: int = 256,
) -> Dict:
    """
    Run PCA + K-Means + clustering metrics for a single iteration snapshot.
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

    # Pairwise dispersion (cosine distance on original normalized embeddings)
    pairwise = pdist(embeddings, metric="cosine")
    pairwise_mean = float(np.mean(pairwise))
    pairwise_std = float(np.std(pairwise))

    # PCA reduction
    emb_reduced, pca, explained_var = reduce_embeddings(embeddings, pca_components)

    best_k = None
    best_silhouette = -1.0
    best_labels = None
    best_centroids = None
    k_results = []

    for k in k_values:
        if k >= n_samples:
            continue
        kmeans = KMeans(
            n_clusters=k,
            n_init=20,
            max_iter=500,
            random_state=42,
        )
        labels = kmeans.fit_predict(emb_reduced)

        silhouette = silhouette_score(emb_reduced, labels, metric="cosine")
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
        "pca_components": pca_components if pca is not None else n_dim,
        "pca_explained_variance": explained_var,
        "optimal_k": best_k,
        "best_silhouette": None if best_silhouette < 0 else best_silhouette,
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
) -> Dict[int, Dict]:
    """
    Run clustering analysis across all iterations.
    """
    results = {}
    for iter_idx in sorted(embeddings_by_iter.keys()):
        emb = embeddings_by_iter[iter_idx]
        print(f"\nAnalyzing iteration {iter_idx} with {len(emb)} samples...")
        iter_result = analyze_iteration(emb, k_values, pca_components=pca_components)
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
    return results


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
    axes[0, 0].set_ylabel("Mean Pairwise Cosine Distance")
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
    image_ids: np.ndarray,
    results_by_iter: Dict[int, Dict],
    top_k: int = 5,
) -> None:
    """
    Generate contact sheets showing nearest images to each centroid.
    """
    images_dir = run_dir / "images"
    contact_dir = run_dir / "plots" / "attractor_contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    for iter_idx, result in results_by_iter.items():
        if result.get("centroids") is None or result.get("labels") is None:
            continue

        centroids = np.array(result["centroids"])
        labels = np.array(result["labels"])
        embeddings = embeddings_by_iter[iter_idx]

        # Recompute embeddings in the same space K-Means used so centroid
        # distances are comparable (PCA reduces dimensionality before KMeans).
        n_components = result.get("pca_components", embeddings.shape[1])
        embeddings_reduced, _, _ = reduce_embeddings(embeddings, n_components)
        if embeddings_reduced.shape[1] != centroids.shape[1]:
            print(
                f"  Warning: centroid dimension mismatch at iteration {iter_idx} "
                f"({embeddings_reduced.shape[1]} vs {centroids.shape[1]}). Skipping contact sheets."
            )
            continue

        if len(image_ids) != len(embeddings):
            print(f"  Warning: image_ids length mismatch at iteration {iter_idx}; skipping contact sheets.")
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
                img_id = image_ids[idx]
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


def main():
    parser = argparse.ArgumentParser(description="Run PCA + K-Means analysis on saved CLIP embeddings.")
    parser.add_argument("run_dir", type=Path, help="Run directory containing checkpoint NPZs.")
    parser.add_argument(
        "--k-values",
        type=str,
        default="5,10,15,20,25,30,35,40,45,50",
        help="Comma-separated list of k values to evaluate.",
    )
    parser.add_argument("--pca-components", type=int, default=256, help="Number of PCA components before K-Means.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for JSON results (default: run_dir/kmeans_analysis.json).",
    )
    parser.add_argument("--contact-top-k", type=int, default=5, help="Top-k nearest images per centroid for contact sheets.")

    args = parser.parse_args()
    run_dir: Path = args.run_dir
    k_values = parse_k_values(args.k_values)
    output_path = args.output or (run_dir / "kmeans_analysis.json")

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print(f"Loading checkpoints from {run_dir} ...")
    embeddings_by_iter, image_ids = load_all_checkpoints(run_dir)
    print(f"Loaded embeddings for iterations: {sorted(embeddings_by_iter.keys())}")
    print(f"Total images covered: {len(np.unique(image_ids))}")

    results = run_clustering_analysis(embeddings_by_iter, k_values=k_values, pca_components=args.pca_components)

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_convergence_metrics(results, plots_dir)
    print_summary_statistics(results)

    # Save centroids/labels for downstream probes
    for iter_idx, result in results.items():
        if result.get("centroids") is None or result.get("labels") is None:
            continue
        np.savez(
            run_dir / f"cluster_data_iter{iter_idx}.npz",
            centroids=np.array(result["centroids"]),
            labels=np.array(result["labels"]),
        )

    generate_contact_sheets(
        run_dir=run_dir,
        embeddings_by_iter=embeddings_by_iter,
        image_ids=image_ids,
        results_by_iter=results,
        top_k=args.contact_top_k,
    )

    output = {
        "run_dir": str(run_dir.resolve()),
        "k_values": k_values,
        "pca_components": args.pca_components,
        "results_by_iteration": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved analysis results to {output_path}")


if __name__ == "__main__":
    main()
