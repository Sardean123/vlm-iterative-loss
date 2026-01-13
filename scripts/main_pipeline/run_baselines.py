import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

from scripts.main_pipeline.test_pipeline_lambda import (
    run_baseline_experiments,
    load_sd_pipeline,
    load_sd_img2img_pipeline,
    BASELINE_NUM_ITERATIONS,
    BASELINE_SD_STRENGTH,
    BASELINE_RANDOM_SEED,
    CLIP_MODEL_ID,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Existing run directory containing the iteration_results CSV")
    ap.add_argument("--csv-name", default=None, help="Override CSV filename if needed")
    ap.add_argument("--num-images", type=int, default=100, help="How many images to run baselines on")
    ap.add_argument("--dataset-split", default="val[:1000]", help="HF split to load from COCO")
    ap.add_argument("--sd-device", default=None, help="Override SD/CLIP device (default: auto cuda->cpu)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    csv_path = run_dir / (args.csv_name or f"{run_dir.name}_iteration_results.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV at {csv_path}")

    df = pd.read_csv(csv_path)
    captions_by_image = {
        int(r.image_id): r.caption
        for r in df[df.iteration == 0].itertuples()
        if isinstance(r.caption, str) and r.caption.strip()
    }
    if not captions_by_image:
        raise RuntimeError("No iteration-0 captions found; cannot run baselines.")

    device = args.sd_device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset and ensure RGB
    dataset = load_dataset("detection-datasets/coco", split=args.dataset_split)
    dataset = dataset.map(lambda ex: {"image": ex["image"].convert("RGB")})

    # Models
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=clip_dtype).to(device)
    clip_model.eval()

    sd_pipe = load_sd_pipeline(device=device)
    try:
        sd_img2img_pipe = load_sd_img2img_pipeline(device=device)
    except Exception as e:
        print(f"img2img unavailable; SD-only baseline skipped: {e}")
        sd_img2img_pipe = None

    outputs = run_baseline_experiments(
        dataset=dataset,
        clip_model=clip_model,
        clip_processor=clip_processor,
        sd_pipe=sd_pipe,
        sd_img2img_pipe=sd_img2img_pipe,
        captions_by_image=captions_by_image,
        num_images=args.num_images,
        num_iterations=BASELINE_NUM_ITERATIONS,
        strength=BASELINE_SD_STRENGTH,
        random_seed=BASELINE_RANDOM_SEED,
    )

    serialized = {}
    for name, runs in outputs.items():
        arrs = []
        for traj in runs:
            if len(traj):
                arrs.append(np.stack(traj, axis=0))
            else:
                arrs.append(np.empty((0, 0), dtype=np.float32))
        serialized[name] = np.array(arrs, dtype=object)

    np.savez_compressed(run_dir / "baseline_clip_embeddings.npz", **serialized)
    meta = {
        "summary": {name: len(runs) for name, runs in outputs.items()},
        "config": {
            "num_images": args.num_images,
            "num_iterations": BASELINE_NUM_ITERATIONS,
            "sd_strength": BASELINE_SD_STRENGTH,
            "random_seed": BASELINE_RANDOM_SEED,
            "dataset_split": args.dataset_split,
        },
        "paths": {"run_dir": str(run_dir.resolve()), "csv": str(csv_path.resolve())},
        "device": device,
    }
    with open(run_dir / "baseline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✓ Baselines saved to", run_dir)

if __name__ == "__main__":
    main()
