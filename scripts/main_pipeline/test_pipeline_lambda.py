import os
import gc
import csv
import json
import torch
import numpy as np
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration, CLIPProcessor, CLIPModel
from transformers.utils import logging as hf_logging
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
from bert_score import BERTScorer
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
from datetime import datetime
from pathlib import Path
import re
from typing import List, Tuple


# CONFIG
# Model configuration
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SENTIMENT_MODEL = 'all-MiniLM-L6-v2'
MAX_NEW_TOKENS = 75
SIM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLAVA_DO_SAMPLE = False  # keep decoding deterministic/greedy

# Iteration configuration
NUM_ITERATIONS = 25  # Number of caption->image->caption cycles
NUM_IMAGES = 1000  # Number of images to process
CONVERGENCE_WINDOW = 3
CONVERGENCE_THRESHOLD = 0.95
DIVERGENCE_K = 3  # Number of consecutive non-converged windows after convergence to mark divergence
CHECKPOINT_INTERVAL = 50  # Save CLIP embedding checkpoints every N images

# Experiment/output configuration
EXPERIMENTS_DIR = Path("experiments")
PIPELINE_DIR = EXPERIMENTS_DIR / f"{NUM_ITERATIONS}_iter_{NUM_IMAGES}_img_1.5"
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SD_MODEL_SCHEDULER = DPMSolverMultistepScheduler
SD_INFERENCE_STEPS = 30
SD_GUIDANCE_SCALE = 7.5
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

# Baseline configuration
INCLUDE_BASELINES = True  # Set to True to run SD-only / random / fixed-caption baselines
BASELINE_NUM_IMAGES = 100
BASELINE_NUM_ITERATIONS = NUM_ITERATIONS
BASELINE_SD_STRENGTH = 0.65
BASELINE_RANDOM_SEED = 420

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
hf_logging.set_verbosity_error()

BERT_SCORER = BERTScorer(lang="en", device=SIM_DEVICE)

# UTILITY FUNCTIONS
def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def clear_memory():
    """Clear GPU memory cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def format_run_name(timestamp: str) -> str:
    """Create a readable identifier for this run."""
    return f"{timestamp}_iter{NUM_ITERATIONS:02d}_img{NUM_IMAGES:03d}"


def setup_run_dirs(run_dir: Path) -> Path:
    """Create the directory tree for this run and return the images directory."""
    images_dir = run_dir / "images"
    plots_dir = run_dir / "plots"
    images_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"Images directory: {images_dir}")
    print(f"Plots directory: {plots_dir}")
    return images_dir

def calculate_detailed_metrics(caption_original, caption_current):
    """Calculate multiple fine-grained metrics"""
    
    # 1. BERTScore
    try:
        P, R, F1 = BERT_SCORER.score([caption_current], [caption_original])
        bert_f1 = float(F1[0])
    except Exception as e:
        print(f"    Warning: BERTScore failed: {e}")
        bert_f1 = 0.0
    
    # 2. Jaccard similarity 
    words_orig = set(re.findall(r'\w+', caption_original.lower()))
    words_curr = set(re.findall(r'\w+', caption_current.lower()))
    jaccard = len(words_orig & words_curr) / len(words_orig | words_curr) if (words_orig | words_curr) else 0
    
    # 3. Length ratio (compression indicator)
    length_ratio = len(caption_current) / len(caption_original) if len(caption_original) > 0 else 0
    
    return {
        'bert_f1': bert_f1,
        'jaccard': jaccard,
        'length_ratio': length_ratio
    }


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float((vec_a @ vec_b) / denom) if denom else 0.0


def detect_convergence(
    embeddings: List[np.ndarray],
    window_size: int = CONVERGENCE_WINDOW,
    threshold: float = CONVERGENCE_THRESHOLD,
) -> Tuple[bool, int, str]:
    """
    Detect whether the trajectory represented by embeddings has converged.

    Returns:
        converged (bool): Whether convergence was detected.
        iteration (int): Iteration at which convergence was flagged, or -1.
        attractor_type (str): "fixed_point", "limit_cycle_2", or "none".
    """
    if len(embeddings) < window_size + 1:
        return False, -1, "none"

    recent_sims = []
    for idx in range(len(embeddings) - window_size + 1, len(embeddings)):
        sim = cosine_similarity(embeddings[idx], embeddings[idx - 1])
        recent_sims.append(sim)

    if recent_sims and np.mean(recent_sims) > threshold:
        return True, len(embeddings) - window_size, "fixed_point"

    if len(embeddings) >= 6:
        sim_t_tminus2 = []
        sim_t_tminus1 = []
        for idx in range(len(embeddings) - window_size, len(embeddings), 2):
            if idx - 2 >= 0:
                sim_t_tminus2.append(cosine_similarity(embeddings[idx], embeddings[idx - 2]))
                sim_t_tminus1.append(cosine_similarity(embeddings[idx], embeddings[idx - 1]))
        if (len(sim_t_tminus2) >= 2 
            and np.mean(sim_t_tminus2) > threshold 
            and np.mean(sim_t_tminus1) < threshold):
            return True, len(embeddings) - window_size, "limit_cycle_2"

    return False, -1, "none"


def get_vision_embedding(image: Image.Image, model: LlavaForConditionalGeneration, processor: LlavaProcessor) -> np.ndarray:
    """
    Extract the pooled vision embedding from LLaVA's vision tower.
    """
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise AttributeError("Processor does not expose an image_processor required for raw vision features.")
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(model.device, dtype=torch.float16)
    with torch.no_grad():
        vision_tower = getattr(model, "vision_tower", None)
        if vision_tower is None:
            raise AttributeError("Model does not expose a vision_tower for embedding extraction.")
        vision_outputs = vision_tower(pixel_values)
        if isinstance(vision_outputs, (tuple, list)):
            vision_features = vision_outputs[0] # shape [batch_size, num_vision_tokens, hidden_dim]
        else:
            vision_features = getattr(vision_outputs, "last_hidden_state", None)
        if vision_features is None:
            raise ValueError("Unable to retrieve vision features from vision tower output.")
        pooled = vision_features.mean(dim=1)
    embedding = pooled.squeeze(0).float().cpu().numpy() # 1 dim vector of length [D] (D=embedding dim)
    del inputs, pixel_values, vision_outputs, vision_features, pooled
    return embedding


def compute_caption_log_prob(
    image: Image.Image,
    caption: str,
    prompt: str,
    model: LlavaForConditionalGeneration,
    processor: LlavaProcessor,
    average: bool = True,
) -> float:
    """
    Compute log P(caption | image, prompt) using teacher forcing.

    The `prompt` should match the one used for generation and end with the assistant cue
    (e.g., "...\\nASSISTANT:"). Labels mask out the prompt tokens so loss is only over
    caption tokens. Returns per-token average log-prob by default; set average=False
    for total log-prob.
    """
    # Tokenize prompt alone (to capture exact prefix ids) and prompt+caption consistently.
    prompt_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    prompt_ids = prompt_inputs["input_ids"]

    full_text = f"{prompt}{caption}"
    inputs = processor(text=full_text, images=image, return_tensors="pt").to(model.device)
    full_ids = inputs["input_ids"]

    prompt_len = prompt_ids.shape[1]
    # Ensure the prompt tokens match the prefix of the full sequence; if not, fall back to
    # the longest common prefix to avoid masking caption tokens.
    if not torch.equal(prompt_ids, full_ids[:, :prompt_len]):
        prompt_flat = prompt_ids[0].tolist()
        full_flat = full_ids[0].tolist()
        max_prefix = 0
        for a, b in zip(prompt_flat, full_flat):
            if a == b:
                max_prefix += 1
            else:
                break
        prompt_len = max_prefix

    labels = full_ids.clone()
    labels[:, :prompt_len] = -100  # ignore prompt tokens

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)    # [B, seq_len, Vocab] - logits
        loss = outputs.loss  # mean over unmasked tokens (Cross Entropy Loss: -log P(caption | image, prompt))
        caption_token_count = (labels != -100).sum().item()
        if caption_token_count == 0:
            log_prob = float('nan')
        elif average:
            log_prob = -float(loss)
        else:
            log_prob = -float(loss) * caption_token_count

    del inputs, prompt_inputs, labels, outputs
    return log_prob

def get_clip_embedding(image: Image.Image, clip_model: CLIPModel, clip_processor: CLIPProcessor, device: str = CLIP_DEVICE) -> np.ndarray:
    """
    Compute a normalized CLIP image embedding.
    """
    inputs = clip_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=next(clip_model.parameters()).dtype)  # [B=1, C=3, H, W]
    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=pixel_values)   # shape [batch_size, hidden_dim]
    # Normalize to unit length for cosine distance consistency
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    embedding = image_features.squeeze(0).float().cpu().numpy()
    del inputs, pixel_values, image_features
    return embedding


def load_sd_pipeline(device: str = "cuda") -> StableDiffusionPipeline:
    """
    Load SD 1.5 pipeline with a fast scheduler.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = SD_MODEL_SCHEDULER.from_config(pipe.scheduler.config)
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"      Warning: xFormers attention not enabled ({e})")
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_sd_img2img_pipeline(device: str = "cuda") -> StableDiffusionImg2ImgPipeline:
    """
    Load SD 1.5 img2img pipeline for SD-only baselines.
    """
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = SD_MODEL_SCHEDULER.from_config(pipe.scheduler.config)
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"      Warning: xFormers attention not enabled for img2img ({e})")
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def save_embeddings_checkpoint(
    run_dir: Path,
    batch_start: int,
    batch_end: int,
    embeddings_by_iter: dict,
    clip_image_ids_by_iter: dict,
    image_ids: list,
    metadata: dict,
) -> Path:
    """
    Save per-iteration CLIP embeddings for a batch of images to an NPZ checkpoint.
    """
    stacked = {}
    stacked_ids = {}
    for iter_idx, emb_list in embeddings_by_iter.items():
        if emb_list:
            stacked[iter_idx] = np.stack(emb_list, axis=0)  # [N_images, D]
            ids = clip_image_ids_by_iter.get(iter_idx, [])
            stacked_ids[iter_idx] = np.array(ids, dtype=np.int32)
    if not stacked:
        print(f"      Skipping checkpoint {batch_start}-{batch_end}: no embeddings to save")
        return None  # type: ignore

    checkpoint_dir = run_dir / "CLIP_embeddings"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_imgs{batch_start}-{batch_end}.npz"
    np.savez(
        checkpoint_path,
        embeddings_by_iter=stacked,
        image_ids_by_iter=stacked_ids,
        image_ids=np.array(image_ids, dtype=np.int32),
        metadata=metadata,
    )
    print(f"✓ Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def collect_initial_captions(iteration_rows: List[dict]) -> dict:
    """
    Build a map of image_id -> first-iteration caption from the main pipeline output.
    """
    captions = {}
    for row in iteration_rows:
        if row.get("iteration") == 0 and row.get("caption"):
            captions[int(row["image_id"])] = str(row["caption"])
    return captions


def run_sd_only_rollout(
    image: Image.Image,
    image_id: int,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    sd_img2img_pipe: StableDiffusionImg2ImgPipeline,
    num_iterations: int,
    strength: float,
) -> List[np.ndarray]:
    """
    Iterate SD without any VLM prompting (img2img), returning CLIP embeddings per step.
    """
    if sd_img2img_pipe is None:
        return []
    current = image.convert("RGB") if image.mode != "RGB" else image
    clip_traj: List[np.ndarray] = []
    for iter_idx in range(num_iterations):
        clip_traj.append(get_clip_embedding(current, clip_model, clip_processor).astype(np.float32))
        if iter_idx == num_iterations - 1:
            break
        generator = torch.Generator(device=sd_img2img_pipe.device).manual_seed((image_id * 1000) + (iter_idx + 1))
        out = sd_img2img_pipe(
            prompt="",
            image=current,
            strength=strength,
            guidance_scale=SD_GUIDANCE_SCALE,
            num_inference_steps=SD_INFERENCE_STEPS,
            generator=generator,
        )
        current = out.images[0]
    return clip_traj


def run_caption_rollout(
    image: Image.Image,
    image_id: int,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    sd_pipe: StableDiffusionPipeline,
    num_iterations: int,
    caption_sequence,
) -> List[np.ndarray]:
    """
    Iterate SD using a provided caption (string or list) without additional VLM calls.
    """
    current = image
    clip_traj: List[np.ndarray] = []
    for iter_idx in range(num_iterations):
        clip_traj.append(get_clip_embedding(current, clip_model, clip_processor).astype(np.float32))
        if iter_idx == num_iterations - 1:
            break

        if isinstance(caption_sequence, list):
            if not caption_sequence:
                break
            caption = caption_sequence[min(iter_idx, len(caption_sequence) - 1)]
        else:
            caption = caption_sequence

        if caption is None:
            break

        generator = torch.Generator(device=sd_pipe.device).manual_seed((image_id * 1000) + (iter_idx + 1))
        out = sd_pipe(
            prompt=caption,
            num_inference_steps=SD_INFERENCE_STEPS,
            guidance_scale=SD_GUIDANCE_SCALE,
            generator=generator,
        )
        current = out.images[0]
    return clip_traj


def run_baseline_experiments(
    dataset,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    sd_pipe: StableDiffusionPipeline,
    sd_img2img_pipe: StableDiffusionImg2ImgPipeline,
    captions_by_image: dict,
    num_images: int,
    num_iterations: int,
    strength: float,
    random_seed: int = 0,
):
    """
    Run SD-only, random-caption, and fixed-caption baselines on a subset of the dataset.
    """
    rng = np.random.default_rng(random_seed)
    caption_pool = list(captions_by_image.values())
    baselines = {"sd_only": [], "random_caption": [], "fixed_caption": []}
    total = min(num_images, len(dataset))

    for idx in range(total):
        image = dataset[idx]["image"]
        image_id = idx + 1
        fixed_caption = captions_by_image.get(image_id)

        if sd_img2img_pipe is not None:
            baselines["sd_only"].append(
                run_sd_only_rollout(
                    image=image,
                    image_id=image_id,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    sd_img2img_pipe=sd_img2img_pipe,
                    num_iterations=num_iterations,
                    strength=strength,
                )
            )

        # Random captions sampled with replacement from the pool of initial captions
        random_seq = rng.choice(caption_pool, size=num_iterations, replace=True).tolist() if caption_pool else []
        baselines["random_caption"].append(
            run_caption_rollout(
                image=image,
                image_id=image_id,
                clip_model=clip_model,
                clip_processor=clip_processor,
                sd_pipe=sd_pipe,
                num_iterations=num_iterations,
                caption_sequence=random_seq,
            )
        )

        if fixed_caption is not None:
            baselines["fixed_caption"].append(
                run_caption_rollout(
                    image=image,
                    image_id=image_id,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    sd_pipe=sd_pipe,
                    num_iterations=num_iterations,
                    caption_sequence=fixed_caption,
                )
            )
        else:
            baselines["fixed_caption"].append([])

    return baselines


# MAIN PIPELINE
def run_multiple_iterations(
    image,
    image_id,
    model,
    processor,
    sim_model,
    clip_model,
    clip_processor,
    sd_pipe,
    prompt,
    timestamp,
    num_iterations,
    images_dir: Path,
    file_prefix: str,
):
    """Run multiple caption->generate->caption cycles"""
    
    iteration_results = []
    current_image = image
    reference_image = image.copy()
    original_caption = None
    original_embedding = None
    original_embedding_norm = None
    prev_embedding = None
    prev_embedding_norm = None
    vision_embeddings: List[np.ndarray] = []
    vision_original_embedding = None
    vision_original_norm = None
    vision_prev_embedding = None
    vision_prev_norm = None
    clip_embeddings = {}
    clip_embeddings_seq: List[np.ndarray] = []
    clip_original_embedding = None
    clip_original_norm = None
    clip_prev_embedding = None
    clip_prev_norm = None
    vision_first_conv_iter = None
    vision_first_attr_type = "none"
    vision_ever_converged = False
    vision_diverged_after = False
    vision_fail_streak = 0
    clip_first_conv_iter = None
    clip_first_attr_type = "none"
    clip_ever_converged = False
    clip_diverged_after = False
    clip_fail_streak = 0
    
    for iteration in range(num_iterations):
        print(f"    Iteration {iteration + 1}/{num_iterations}")
        
        result = {
            'image_id': image_id,
            'iteration': iteration,
            'caption': None,
            'similarity_to_original': None,
            'similarity_to_previous': None,
            'bert_f1': None,
            'jaccard': None,
            'length_ratio': None,
            'vision_embedding_norm': None,
            'vision_sim_to_original': None,
            'vision_sim_to_previous': None,
            'clip_embedding_norm': None,
            'clip_sim_to_original': None,
            'clip_sim_to_previous': None,
            'log_prob_original_caption': None,
            'log_prob_caption_on_original_image': None,
            'vision_currently_converged': False,
            'vision_ever_converged': False,
            'vision_diverged_after_convergence': False,
            'vision_converged': False,
            'vision_convergence_iteration': -1,
            'vision_attractor_type': "none",
            'clip_currently_converged': False,
            'clip_ever_converged': False,
            'clip_diverged_after_convergence': False,
            'clip_converged': False,
            'clip_convergence_iteration': -1,
            'clip_attractor_type': "none",
            'error': None
        }
        
        try:
            # Save image
            if iteration == 0:
                img_path = images_dir / f"{file_prefix}_img{image_id:03d}_iter{iteration}_original.png"
            else:
                img_path = images_dir / f"{file_prefix}_img{image_id:03d}_iter{iteration}_generated.png"
            current_image.save(img_path)

            # Extract CLIP embedding for current image
            added_clip_embedding = False
            try:
                clip_emb = get_clip_embedding(current_image, clip_model, clip_processor)
                clip_embeddings[iteration] = clip_emb
                clip_norm = np.linalg.norm(clip_emb)
                result['clip_embedding_norm'] = float(clip_norm)
                if clip_original_embedding is None:
                    clip_original_embedding = clip_emb
                    clip_original_norm = clip_norm
                    result['clip_sim_to_original'] = 1.0
                else:
                    denom_orig = clip_original_norm * clip_norm
                    sim_orig = float((clip_original_embedding @ clip_emb) / denom_orig) if denom_orig else 0.0
                    result['clip_sim_to_original'] = sim_orig
                if clip_prev_embedding is None:
                    result['clip_sim_to_previous'] = 1.0
                else:
                    denom_prev = clip_prev_norm * clip_norm
                    sim_prev = float((clip_prev_embedding @ clip_emb) / denom_prev) if denom_prev else 0.0
                    result['clip_sim_to_previous'] = sim_prev
                clip_prev_embedding = clip_emb
                clip_prev_norm = clip_norm
                clip_embeddings_seq.append(clip_emb)
                added_clip_embedding = True
            except Exception as clip_error:
                print(f"      Warning: CLIP embedding failed: {clip_error}")

            # Extract vision embedding for current image
            added_vision_embedding = False
            try:
                vision_emb = get_vision_embedding(current_image, model, processor)
                vision_embeddings.append(vision_emb)
                vision_norm = np.linalg.norm(vision_emb)
                result['vision_embedding_norm'] = float(vision_norm)
                if vision_original_embedding is None:
                    vision_original_embedding = vision_emb
                    vision_original_norm = vision_norm
                    result['vision_sim_to_original'] = 1.0
                else:
                    denom_orig = vision_original_norm * vision_norm
                    sim_orig = float((vision_original_embedding @ vision_emb) / denom_orig) if denom_orig else 0.0
                    result['vision_sim_to_original'] = sim_orig
                if vision_prev_embedding is None:
                    result['vision_sim_to_previous'] = 1.0
                else:
                    denom_prev = vision_prev_norm * vision_norm
                    sim_prev = float((vision_prev_embedding @ vision_emb) / denom_prev) if denom_prev else 0.0
                    result['vision_sim_to_previous'] = sim_prev
                vision_prev_embedding = vision_emb
                vision_prev_norm = vision_norm
                added_vision_embedding = True
            except Exception as vision_error:
                print(f"      Warning: vision embedding failed: {vision_error}")

            # Log probability of the original caption given current image (skip first/original image)
            if iteration > 0 and original_caption:
                try:
                    log_prob_original = compute_caption_log_prob(current_image, original_caption, prompt, model, processor)
                    result['log_prob_original_caption'] = log_prob_original
                    # print(f"      Log P(original_caption | current_image): {log_prob_original:.3f}")
                except Exception as logprob_error:
                    print(f"      Warning: log-prob computation failed: {logprob_error}")
                finally:
                    clear_memory()

            # Detect convergence based on accumulated vision embeddings
            if added_vision_embedding and vision_embeddings:
                converged, _, attr_type = detect_convergence(vision_embeddings)
                result["vision_currently_converged"] = converged

                if converged:
                    vision_fail_streak = 0
                    if not vision_ever_converged:
                        if attr_type == "limit_cycle_2":
                            stable_start = iteration - 2 * (CONVERGENCE_WINDOW - 1)
                        else:
                            stable_start = iteration - (CONVERGENCE_WINDOW - 1)
                        vision_first_conv_iter = max(0, stable_start)
                        vision_first_attr_type = attr_type
                        vision_ever_converged = True
                else:
                    if vision_ever_converged and not vision_diverged_after:
                        vision_fail_streak += 1
                        if vision_fail_streak >= DIVERGENCE_K:
                            vision_diverged_after = True

                result["vision_ever_converged"] = vision_ever_converged
                result["vision_diverged_after_convergence"] = vision_diverged_after
                result['vision_convergence_iteration'] = vision_first_conv_iter if vision_first_conv_iter is not None else -1
                result['vision_attractor_type'] = vision_first_attr_type
                result['vision_converged'] = vision_ever_converged

                if converged and iteration < num_iterations - 1:
                    print(f"      CONVERGED to {vision_first_attr_type} at iteration {result['vision_convergence_iteration']}!")
            else:
                # Propagate sticky state even if we could not evaluate this iteration
                result["vision_currently_converged"] = False
                result["vision_ever_converged"] = vision_ever_converged
                result["vision_diverged_after_convergence"] = vision_diverged_after
                result['vision_convergence_iteration'] = vision_first_conv_iter if vision_first_conv_iter is not None else -1
                result['vision_attractor_type'] = vision_first_attr_type
                result['vision_converged'] = vision_ever_converged

            # Detect convergence based on accumulated CLIP embeddings
            if added_clip_embedding and clip_embeddings_seq:
                clip_conv, _, clip_attr = detect_convergence(clip_embeddings_seq)
                result['clip_currently_converged'] = clip_conv

                if clip_conv:
                    clip_fail_streak = 0
                    if not clip_ever_converged:
                        if clip_attr == "limit_cycle_2":
                            stable_start = iteration - 2 * (CONVERGENCE_WINDOW - 1)
                        else:
                            stable_start = iteration - (CONVERGENCE_WINDOW - 1)
                        clip_first_conv_iter = max(0, stable_start)
                        clip_first_attr_type = clip_attr
                        clip_ever_converged = True
                else:
                    if clip_ever_converged and not clip_diverged_after:
                        clip_fail_streak += 1
                        if clip_fail_streak >= DIVERGENCE_K:
                            clip_diverged_after = True

                result['clip_ever_converged'] = clip_ever_converged
                result['clip_diverged_after_convergence'] = clip_diverged_after
                result['clip_convergence_iteration'] = clip_first_conv_iter if clip_first_conv_iter is not None else -1
                result['clip_attractor_type'] = clip_first_attr_type
                result['clip_converged'] = clip_ever_converged

                if clip_conv and iteration < num_iterations - 1:
                    print(f"      CLIP CONVERGED to {clip_first_attr_type} at iteration {result['clip_convergence_iteration']}!")
            else:
                result['clip_currently_converged'] = False
                result['clip_ever_converged'] = clip_ever_converged
                result['clip_diverged_after_convergence'] = clip_diverged_after
                result['clip_convergence_iteration'] = clip_first_conv_iter if clip_first_conv_iter is not None else -1
                result['clip_attractor_type'] = clip_first_attr_type
                result['clip_converged'] = clip_ever_converged
            
            # Generate caption (deterministic/greedy decoding)
            inputs = processor(text=prompt, images=current_image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=LLAVA_DO_SAMPLE
                )
            
            caption = processor.decode(output[0], skip_special_tokens=True)
            
            # Extract assistant's response
            if "ASSISTANT:" in caption:
                caption = caption.split("ASSISTANT:")[1].strip()
            
            result['caption'] = caption
            print(f"      Caption: {caption[:60]}...")

            # Store original caption
            if iteration == 0:
                original_caption = caption

            # Log probability of the current caption if conditioned on the original image
            try:
                log_prob_caption_on_orig = compute_caption_log_prob(reference_image, caption, prompt, model, processor)
                result['log_prob_caption_on_original_image'] = log_prob_caption_on_orig
                # print(f"      Log P(caption | original_image): {log_prob_caption_on_orig:.3f}")
            except Exception as caption_logprob_error:
                print(f"      Warning: reverse log-prob failed: {caption_logprob_error}")
            finally:
                clear_memory()
            
            # Clear memory
            del inputs, output
            clear_memory()
            
            # Calculate semantic similarity
            emb_current = sim_model.encode(caption)
            emb_current_norm = np.linalg.norm(emb_current)
            if original_embedding is None:
                original_embedding = emb_current
                original_embedding_norm = emb_current_norm
                sim_to_original = 1.0
            else:
                denom = original_embedding_norm * emb_current_norm
                sim_to_original = float((original_embedding @ emb_current) / denom) if denom else 0.0
            result['similarity_to_original'] = sim_to_original

            if prev_embedding is None:
                result['similarity_to_previous'] = 1.0
            else:
                denom_prev = prev_embedding_norm * emb_current_norm
                sim_to_prev = float((prev_embedding @ emb_current) / denom_prev) if denom_prev else 0.0
                result['similarity_to_previous'] = sim_to_prev

            prev_embedding = emb_current
            prev_embedding_norm = emb_current_norm
            
            # Calculate detailed metrics
            metrics = calculate_detailed_metrics(original_caption, caption)
            result.update(metrics)
            
            print(f"      Similarity to original: {sim_to_original:.3f}")
            print(f"      BERTScore F1: {metrics['bert_f1']:.3f}")
            
            # Generate next image (except on last iteration)
            if iteration < num_iterations - 1:
                print(f"      Generating image via SD 1.5 (diffusers)...")
                seed = (image_id * 100) + (iteration + 1)
                generator = torch.Generator(device=sd_pipe.device).manual_seed(seed)
                sd_output = sd_pipe(
                    prompt=caption,
                    num_inference_steps=SD_INFERENCE_STEPS,
                    guidance_scale=SD_GUIDANCE_SCALE,
                    generator=generator,
                )
                current_image = sd_output.images[0]

                clear_memory()
            
        except Exception as e:
            result['error'] = str(e)
            print(f"      Error: {e}")
            # Stop iterations on error
            break
        
        iteration_results.append(result)
    
    return iteration_results, clip_embeddings

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = format_run_name(timestamp)
    run_dir = PIPELINE_DIR / run_name
    images_dir = setup_run_dirs(run_dir)
    plots_dir = run_dir / "plots"
    metadata_path = run_dir / "metadata.json"
    
    print("MULTI-ITERATION IMAGE CAPTIONING PIPELINE")
    print(f"Processing {NUM_IMAGES} images with {NUM_ITERATIONS} iterations each")
    
    # Check GPU availability
    print("\n1. CHECKING GPU AVAILABILITY")
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected!")
        return
    
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print_gpu_memory()
    
    # Load LLaVA Model
    print("\n2. LOADING LLAVA MODEL")
    print("-" * 80)
    
    processor = LlavaProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    print(f"Model loaded successfully")
    print_gpu_memory()
    
    # Load CLIP Model
    print("\n3. LOADING CLIP MODEL")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=clip_torch_dtype).to(CLIP_DEVICE)
    clip_model.eval()
    print("CLIP model loaded")
    print_gpu_memory()
    
    # Load Sentence Transformer
    print("\n4. LOADING SENTENCE TRANSFORMER")
    sim_model = SentenceTransformer(SENTIMENT_MODEL, device=SIM_DEVICE)
    print("Sentence transformer loaded")
    
    # Load Dataset
    print("\n5. LOADING DATASET")
    
    dataset_name = "detection-datasets/coco"
    dataset_split = f"val[:{NUM_IMAGES}]"
    try:
        dataset = load_dataset(dataset_name, split=dataset_split)
        print(f"✓ Loaded {len(dataset)} images from COCO dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    dataset_size = len(dataset)
    baseline_outputs = None
    baseline_path = None
    
    # Load SD pipeline
    print("\n6. LOADING STABLE DIFFUSION 1.5 (diffusers)")
    sd_pipe = load_sd_pipeline(device=SIM_DEVICE if torch.cuda.is_available() else "cpu")
    print("Stable Diffusion pipeline loaded")
    print_gpu_memory()
    
    # Prepare prompt
    prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
    
    # Process images
    print("\n7. PROCESSING IMAGES")
    
    all_results = []
    batch_embeddings_by_iter = {iter_idx: [] for iter_idx in range(NUM_ITERATIONS)}
    batch_clip_ids_by_iter = {iter_idx: [] for iter_idx in range(NUM_ITERATIONS)}
    batch_image_ids = []
    batch_start_idx = 1
    
    for i, sample in enumerate(dataset):
        print(f"\n[{i+1}/{NUM_IMAGES}] Processing image {i+1}...")
        print("-" * 80)
        
        image = sample['image']
        iteration_results, clip_embeddings = run_multiple_iterations(
            image,
            i + 1,
            model,
            processor,
            sim_model,
            clip_model,
            clip_processor,
            sd_pipe,
            prompt,
            timestamp,
            NUM_ITERATIONS,
            images_dir,
            run_name,
        )
        all_results.extend(iteration_results)
        
        # Accumulate CLIP embeddings for checkpointing
        image_id = i + 1
        batch_image_ids.append(image_id)
        for iter_idx, emb in clip_embeddings.items():
            if emb is not None:
                batch_embeddings_by_iter.setdefault(iter_idx, []).append(emb.astype(np.float32))
                batch_clip_ids_by_iter.setdefault(iter_idx, []).append(image_id)
        
        # Save checkpoint on interval or at the end
        if ((i + 1) % CHECKPOINT_INTERVAL == 0) or (i == dataset_size - 1):
            checkpoint_metadata = {
                'batch_start': batch_start_idx,
                'batch_end': image_id,
                'clip_model_id': CLIP_MODEL_ID,
                'normalized': True,
                'num_iterations': NUM_ITERATIONS,
                'created_at': timestamp,
            }
            save_embeddings_checkpoint(
                run_dir,
                batch_start_idx,
                image_id,
                batch_embeddings_by_iter,
                batch_clip_ids_by_iter,
                batch_image_ids,
                checkpoint_metadata,
            )
            batch_embeddings_by_iter = {iter_idx: [] for iter_idx in range(NUM_ITERATIONS)}
            batch_clip_ids_by_iter = {iter_idx: [] for iter_idx in range(NUM_ITERATIONS)}
            batch_image_ids = []
            batch_start_idx = image_id + 1
        
        print_gpu_memory()
        print()
    
    # Save results to CSV
    print("\n8. SAVING RESULTS")
    
    csv_path = run_dir / f"{run_name}_iteration_results.csv"
    
    fieldnames = [
        'image_id', 'iteration', 'caption',
        'similarity_to_original', 'similarity_to_previous',
        'bert_f1', 'jaccard', 'length_ratio',
        'vision_embedding_norm', 'vision_sim_to_original', 'vision_sim_to_previous',
        'clip_embedding_norm', 'clip_sim_to_original', 'clip_sim_to_previous',
        'log_prob_original_caption',
        'log_prob_caption_on_original_image',
        'vision_currently_converged', 'vision_ever_converged', 'vision_diverged_after_convergence',
        'vision_converged', 'vision_convergence_iteration', 'vision_attractor_type',
        'clip_currently_converged', 'clip_ever_converged', 'clip_diverged_after_convergence',
        'clip_converged', 'clip_convergence_iteration', 'clip_attractor_type',
        'error'
    ]
    
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"✓ Results saved to: {csv_path}")
    
    # Calculate statistics
    print("\n9. DECAY ANALYSIS")
    
    # Group by iteration (keep entries without errors)
    by_iteration = {}
    for r in all_results:
        if r['error'] is None:
            iter_num = r['iteration']
            if iter_num not in by_iteration:
                by_iteration[iter_num] = []
            by_iteration[iter_num].append(r)
    
    iteration_summary = []
    decay_stats = None
    fastest = []
    most_stable = []
    fastest_vision = []
    most_stable_vision = []
    fastest_clip = []
    most_stable_clip = []
    if by_iteration:
        def safe_mean(values):
            vals = [v for v in values if v is not None]
            return float(np.mean(vals)) if vals else None

        print("\nAverage Metrics by Iteration (text / vision / CLIP):")
        header = (
            f"{'Iter':<6}"
            f"{'SimO(txt)':<11}{'SimO(vis)':<11}{'SimO(clip)':<12}"
            f"{'SimP(txt)':<11}{'SimP(vis)':<11}{'SimP(clip)':<12}"
            f"{'BERT-F1':<10}{'Jaccard':<10}"
        )
        print(header)
        conv_fraction_summary = []
        
        for iter_num in sorted(by_iteration.keys()):
            results = by_iteration[iter_num]
            avg_sim_orig = safe_mean([r['similarity_to_original'] for r in results])
            avg_sim_prev = safe_mean([r['similarity_to_previous'] for r in results])
            avg_sim_orig_vis = safe_mean([r['vision_sim_to_original'] for r in results])
            avg_sim_prev_vis = safe_mean([r['vision_sim_to_previous'] for r in results])
            avg_sim_orig_clip = safe_mean([r['clip_sim_to_original'] for r in results])
            avg_sim_prev_clip = safe_mean([r['clip_sim_to_previous'] for r in results])
            avg_bert = safe_mean([r['bert_f1'] for r in results])
            avg_jaccard = safe_mean([r['jaccard'] for r in results])
            denom = len(results)
            vis_frac_ever = (sum(1 for r in results if r.get('vision_ever_converged')) / denom) if denom else None
            vis_frac_diverged = (sum(1 for r in results if r.get('vision_diverged_after_convergence')) / denom) if denom else None
            vis_frac_current = (sum(1 for r in results if r.get('vision_currently_converged')) / denom) if denom else None
            clip_frac_ever = (sum(1 for r in results if r.get('clip_ever_converged')) / denom) if denom else None
            clip_frac_diverged = (sum(1 for r in results if r.get('clip_diverged_after_convergence')) / denom) if denom else None
            clip_frac_current = (sum(1 for r in results if r.get('clip_currently_converged')) / denom) if denom else None
            
            print(
                f"{iter_num:<6}"
                f"{(avg_sim_orig if avg_sim_orig is not None else float('nan')):<11.3f}"
                f"{(avg_sim_orig_vis if avg_sim_orig_vis is not None else float('nan')):<11.3f}"
                f"{(avg_sim_orig_clip if avg_sim_orig_clip is not None else float('nan')):<12.3f}"
                f"{(avg_sim_prev if avg_sim_prev is not None else float('nan')):<11.3f}"
                f"{(avg_sim_prev_vis if avg_sim_prev_vis is not None else float('nan')):<11.3f}"
                f"{(avg_sim_prev_clip if avg_sim_prev_clip is not None else float('nan')):<12.3f}"
                f"{(avg_bert if avg_bert is not None else float('nan')):<10.3f}"
                f"{(avg_jaccard if avg_jaccard is not None else float('nan')):<10.3f}"
            )
            conv_fraction_summary.append({
                'iteration': iter_num,
                'vision_ever': vis_frac_ever,
                'vision_diverged': vis_frac_diverged,
                'vision_current': vis_frac_current,
                'clip_ever': clip_frac_ever,
                'clip_diverged': clip_frac_diverged,
                'clip_current': clip_frac_current,
            })
            iteration_summary.append({
                'iteration': iter_num,
                'similarity_to_original_mean': float(avg_sim_orig) if avg_sim_orig is not None else None,
                'similarity_to_previous_mean': float(avg_sim_prev) if avg_sim_prev is not None else None,
                'vision_sim_to_original_mean': float(avg_sim_orig_vis) if avg_sim_orig_vis is not None else None,
                'vision_sim_to_previous_mean': float(avg_sim_prev_vis) if avg_sim_prev_vis is not None else None,
                'clip_sim_to_original_mean': float(avg_sim_orig_clip) if avg_sim_orig_clip is not None else None,
                'clip_sim_to_previous_mean': float(avg_sim_prev_clip) if avg_sim_prev_clip is not None else None,
                'bert_f1_mean': float(avg_bert) if avg_bert is not None else None,
                'jaccard_mean': float(avg_jaccard) if avg_jaccard is not None else None,
                'vision_fraction_ever_converged': float(vis_frac_ever) if vis_frac_ever is not None else None,
                'vision_fraction_diverged_after_convergence': float(vis_frac_diverged) if vis_frac_diverged is not None else None,
                'vision_fraction_currently_converged': float(vis_frac_current) if vis_frac_current is not None else None,
                'clip_fraction_ever_converged': float(clip_frac_ever) if clip_frac_ever is not None else None,
                'clip_fraction_diverged_after_convergence': float(clip_frac_diverged) if clip_frac_diverged is not None else None,
                'clip_fraction_currently_converged': float(clip_frac_current) if clip_frac_current is not None else None,
            })
        
        if conv_fraction_summary:
            print("\nConvergence Fractions by Iteration (vision / clip):")
            conv_header = (
                f"{'Iter':<6}"
                f"{'Vis-ever':<11}{'Vis-div':<11}{'Vis-now':<11}"
                f"{'Clip-ever':<11}{'Clip-div':<11}{'Clip-now':<11}"
            )
            print(conv_header)

            def fmt_frac(val):
                return f"{val:.3f}" if val is not None else "nan"

            for stats in conv_fraction_summary:
                print(
                    f"{stats['iteration']:<6}"
                    f"{fmt_frac(stats['vision_ever']):<11}"
                    f"{fmt_frac(stats['vision_diverged']):<11}"
                    f"{fmt_frac(stats['vision_current']):<11}"
                    f"{fmt_frac(stats['clip_ever']):<11}"
                    f"{fmt_frac(stats['clip_diverged']):<11}"
                    f"{fmt_frac(stats['clip_current']):<11}"
                )
        
        # Calculate decay rate
        if len(by_iteration) >= 2:
            print(f"\nDecay Statistics (iteration 0 baseline):")

            def compute_decay(metric_key: str, label: str):
                last_iter_idx = max(by_iteration.keys())
                start_vals = [r[metric_key] for r in by_iteration.get(0, []) if r[metric_key] is not None]
                end_vals = [r[metric_key] for r in by_iteration.get(last_iter_idx, []) if r[metric_key] is not None]
                if not start_vals or not end_vals:
                    print(f"  {label}: insufficient data")
                    return None
                start_mean = float(np.mean(start_vals))
                end_mean = float(np.mean(end_vals))
                total_decay = start_mean - end_mean
                avg_decay = total_decay / last_iter_idx if last_iter_idx > 0 else 0.0
                print(f"  {label}:")
                print(f"    Initial (iter 0): {start_mean:.3f}")
                print(f"    Final   (iter {last_iter_idx}): {end_mean:.3f}")
                print(f"    Total decay: {total_decay:.3f}")
                print(f"    Avg decay/iter: {avg_decay:.3f}")
                return {
                    'initial_similarity': start_mean,
                    'final_similarity': end_mean,
                    'total_decay': total_decay,
                    'avg_decay_per_iteration': avg_decay,
                    'last_iteration': int(last_iter_idx),
                }

            decay_stats = {
                'text': compute_decay('similarity_to_original', 'Text (sentence)'),
                'vision': compute_decay('vision_sim_to_original', 'Vision tower'),
                'clip': compute_decay('clip_sim_to_original', 'CLIP'),
            }
            # Backward-compatible aliases for text decay
            if decay_stats['text']:
                decay_stats.update({
                    'initial_similarity': float(decay_stats['text']['initial_similarity']),
                    'final_similarity': float(decay_stats['text']['final_similarity']),
                    'total_decay': float(decay_stats['text']['total_decay']),
                    'avg_decay_per_iteration': float(decay_stats['text']['avg_decay_per_iteration']),
                    'last_iteration': int(decay_stats['text']['last_iteration']),
                })
        
        # Find extreme cases
        print(f"\nFastest Degrading Images (text, lowest final similarity):")
        final_iter = max(by_iteration.keys())
        final_results_all = [r for r in by_iteration[final_iter] if r['similarity_to_original'] is not None]
        final_results_vis_all = [r for r in by_iteration[final_iter] if r['vision_sim_to_original'] is not None]
        final_results_clip_all = [r for r in by_iteration[final_iter] if r['clip_sim_to_original'] is not None]
        if final_results_all:
            final_results = sorted(final_results_all, key=lambda x: x['similarity_to_original'])[:5]
            for r in final_results:
                print(f"  Image {r['image_id']}: {r['similarity_to_original']:.3f}")
            fastest = [
                {'image_id': int(r['image_id']), 'similarity_to_original': float(r['similarity_to_original'])}
                for r in final_results
            ]
        else:
            print("  No data available.")

        print(f"\nFastest Degrading Images (vision, lowest final similarity):")
        if final_results_vis_all:
            final_results_vis = sorted(final_results_vis_all, key=lambda x: x['vision_sim_to_original'])[:5]
            for r in final_results_vis:
                print(f"  Image {r['image_id']}: {r['vision_sim_to_original']:.3f}")
            fastest_vision = [
                {'image_id': int(r['image_id']), 'vision_sim_to_original': float(r['vision_sim_to_original'])}
                for r in final_results_vis
            ]
        else:
            print("  No data available.")

        print(f"\nFastest Degrading Images (CLIP, lowest final similarity):")
        if final_results_clip_all:
            final_results_clip = sorted(final_results_clip_all, key=lambda x: x['clip_sim_to_original'])[:5]
            for r in final_results_clip:
                print(f"  Image {r['image_id']}: {r['clip_sim_to_original']:.3f}")
            fastest_clip = [
                {'image_id': int(r['image_id']), 'clip_sim_to_original': float(r['clip_sim_to_original'])}
                for r in final_results_clip
            ]
        else:
            print("  No data available.")
        
        print(f"\nMost Stable Images (text, highest final similarity):")
        if final_results_all:
            stable_results = sorted(final_results_all, key=lambda x: x['similarity_to_original'], reverse=True)[:5]
            for r in stable_results:
                print(f"  Image {r['image_id']}: {r['similarity_to_original']:.3f}")
            most_stable = [
                {'image_id': int(r['image_id']), 'similarity_to_original': float(r['similarity_to_original'])}
                for r in stable_results
            ]
        else:
            print("  No data available.")

        print(f"\nMost Stable Images (vision, highest final similarity):")
        if final_results_vis_all:
            stable_results_vis = sorted(final_results_vis_all, key=lambda x: x['vision_sim_to_original'], reverse=True)[:5]
            for r in stable_results_vis:
                print(f"  Image {r['image_id']}: {r['vision_sim_to_original']:.3f}")
            most_stable_vision = [
                {'image_id': int(r['image_id']), 'vision_sim_to_original': float(r['vision_sim_to_original'])}
                for r in stable_results_vis
            ]
        else:
            print("  No data available.")

        print(f"\nMost Stable Images (CLIP, highest final similarity):")
        if final_results_clip_all:
            stable_results_clip = sorted(final_results_clip_all, key=lambda x: x['clip_sim_to_original'], reverse=True)[:5]
            for r in stable_results_clip:
                print(f"  Image {r['image_id']}: {r['clip_sim_to_original']:.3f}")
            most_stable_clip = [
                {'image_id': int(r['image_id']), 'clip_sim_to_original': float(r['clip_sim_to_original'])}
                for r in stable_results_clip
            ]
        else:
            print("  No data available.")

    baseline_summary = None
    if INCLUDE_BASELINES:
        print("\n10. RUNNING BASELINE EXPERIMENTS")
        captions_by_image = collect_initial_captions(all_results)
        if not captions_by_image:
            print("  ⚠️  No initial captions found; skipping baselines.")
        else:
            try:
                sd_img2img_pipe = load_sd_img2img_pipeline(device=sd_pipe.device)
            except Exception as e:
                print(f"  Warning: could not load SD img2img pipeline ({e}); SD-only baseline will be empty.")
                sd_img2img_pipe = None

            baseline_outputs = run_baseline_experiments(
                dataset=dataset,
                clip_model=clip_model,
                clip_processor=clip_processor,
                sd_pipe=sd_pipe,
                sd_img2img_pipe=sd_img2img_pipe,
                captions_by_image=captions_by_image,
                num_images=BASELINE_NUM_IMAGES,
                num_iterations=BASELINE_NUM_ITERATIONS,
                strength=BASELINE_SD_STRENGTH,
                random_seed=BASELINE_RANDOM_SEED,
            )

            baseline_path = run_dir / "baseline_clip_embeddings.npz"
            serialized = {}
            for name, runs in baseline_outputs.items():
                stacked_runs = []
                for traj in runs:
                    if len(traj):
                        stacked_runs.append(np.stack(traj, axis=0))
                    else:
                        stacked_runs.append(np.empty((0, 0), dtype=np.float32))
                serialized[name] = np.array(stacked_runs, dtype=object)

            if serialized:
                np.savez_compressed(baseline_path, **serialized)
                print(f"✓ Baseline CLIP embeddings saved to: {baseline_path}")
                baseline_summary = {name: len(runs) for name, runs in baseline_outputs.items()}
            else:
                print("  ⚠️  Baseline outputs empty; nothing saved.")

    metadata = {
        'run_name': run_name,
        'timestamp': timestamp,
        'config': {
            'model_id': MODEL_ID,
            'sentence_transformer': SENTIMENT_MODEL,
            'max_new_tokens': MAX_NEW_TOKENS,
            'llava_do_sample': LLAVA_DO_SAMPLE,
            'num_iterations': NUM_ITERATIONS,
            'num_images_requested': NUM_IMAGES,
            'sd_model_id': SD_MODEL_ID,
            'sd_inference_steps': SD_INFERENCE_STEPS,
            'sd_guidance_scale': SD_GUIDANCE_SCALE,
            'convergence_window': CONVERGENCE_WINDOW,
            'convergence_threshold': CONVERGENCE_THRESHOLD,
            'divergence_k': DIVERGENCE_K,
            'checkpoint_interval': CHECKPOINT_INTERVAL,
        },
        'dataset': {
            'name': dataset_name,
            'split': dataset_split,
            'num_loaded': len(dataset),
        },
        'paths': {
            'run_dir': str(run_dir.resolve()),
            'images_dir': str(images_dir.resolve()),
            'plots_dir': str(plots_dir.resolve()),
            'results_csv': str(csv_path.resolve()),
        },
        'results': {
            'rows_saved': len(all_results),
            'iteration_summary': iteration_summary,
            'decay_stats': decay_stats,
            'fastest_degrading': fastest,
            'most_stable': most_stable,
            'fastest_degrading_vision': fastest_vision,
            'most_stable_vision': most_stable_vision,
            'fastest_degrading_clip': fastest_clip,
            'most_stable_clip': most_stable_clip,
        },
        'embeddings': {
            'clip_model_id': CLIP_MODEL_ID,
            'clip_device': CLIP_DEVICE,
            'clip_normalized': True,
        },
        'baselines': {
            'enabled': INCLUDE_BASELINES,
            'num_images': BASELINE_NUM_IMAGES,
            'num_iterations': BASELINE_NUM_ITERATIONS,
            'sd_strength': BASELINE_SD_STRENGTH,
            'random_seed': BASELINE_RANDOM_SEED,
            'summary': baseline_summary,
            'embeddings_path': str(baseline_path.resolve()) if baseline_path and baseline_summary else None,
        }
    }

    with metadata_path.open('w', encoding='utf-8') as meta_file:
        json.dump(metadata, meta_file, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    print("MULTI-ITERATION PROCESSING COMPLETE")
    print(f"\nOutputs saved to: {images_dir}/")
    print(f"CSV results: {csv_path}")
    print(f"Metadata: {metadata_path}")
    print(f"\nTotal iterations completed: {len(all_results)}")
    print(f"\nFinal GPU Memory Usage:")
    print_gpu_memory()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        clear_memory()
        print("Done!")
