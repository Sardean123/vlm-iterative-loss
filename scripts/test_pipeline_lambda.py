import os
import gc
import csv
import json
import torch
import numpy as np
import requests
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
import replicate
from datetime import datetime
from pathlib import Path
import re
from typing import List, Tuple


# CONFIG
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN not found in environment variables")
os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN

# Model configuration
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SENTIMENT_MODEL = 'all-MiniLM-L6-v2'
MAX_NEW_TOKENS = 200
SIM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLAVA_DO_SAMPLE = False  # keep decoding deterministic/greedy

# Iteration configuration
NUM_ITERATIONS = 25  # Number of caption->image->caption cycles
NUM_IMAGES = 50  # Number of images to process
CONVERGENCE_WINDOW = 3
CONVERGENCE_THRESHOLD = 0.99

# Experiment/output configuration
EXPERIMENTS_DIR = Path("experiments")
PIPELINE_DIR = EXPERIMENTS_DIR / f"{NUM_ITERATIONS}_iter_11.18"
SDXL_MODEL_ID = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"


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
        P, R, F1 = bert_score([caption_current], [caption_original], lang="en", verbose=False)
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
        iteration (int): Zero-based iteration at which convergence was flagged, or -1.
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
        cycle_sims = []
        for idx in range(len(embeddings) - 4, len(embeddings), 2):
            if idx - 2 >= 0:
                cycle_sims.append(cosine_similarity(embeddings[idx], embeddings[idx - 2]))
        if len(cycle_sims) >= 2 and np.mean(cycle_sims) > threshold:
            return True, len(embeddings) - 4, "limit_cycle_2"

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
            vision_features = vision_outputs[0]
        else:
            vision_features = getattr(vision_outputs, "last_hidden_state", None)
        if vision_features is None:
            raise ValueError("Unable to retrieve vision features from vision tower output.")
        pooled = vision_features.mean(dim=1)
    embedding = pooled.squeeze(0).float().cpu().numpy()
    del inputs, pixel_values, vision_outputs, vision_features, pooled
    return embedding


def compute_caption_log_prob(image: Image.Image, caption: str, model: LlavaForConditionalGeneration, processor: LlavaProcessor) -> float:
    """
    Compute log P(caption | image) by running LLaVA with labels equal to the prompt tokens.
    """
    conditioned_prompt = f"USER: <image>\n{caption}\nASSISTANT:"
    inputs = processor(text=conditioned_prompt, images=image, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
        log_prob = -float(outputs.loss)
    del inputs, outputs
    return log_prob

# MAIN PIPELINE
def run_multiple_iterations(image, image_id, model, processor, sim_model, prompt, timestamp, num_iterations, images_dir: Path, file_prefix: str):
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
    
    for iteration in range(num_iterations):
        print(f"    Iteration {iteration + 1}/{num_iterations}")
        
        result = {
            'image_id': image_id,
            'iteration': iteration + 1,
            'caption': None,
            'similarity_to_original': None,
            'similarity_to_previous': None,
            'bert_f1': None,
            'jaccard': None,
            'length_ratio': None,
            'vision_embedding_norm': None,
            'vision_sim_to_original': None,
            'vision_sim_to_previous': None,
            'log_prob_original_caption': None,
            'log_prob_caption_on_original_image': None,
            'converged': False,
            'convergence_iteration': -1,
            'attractor_type': "none",
            'error': None
        }
        
        try:
            # Save image
            if iteration == 0:
                img_path = images_dir / f"{file_prefix}_img{image_id:03d}_iter{iteration}_original.png"
            else:
                img_path = images_dir / f"{file_prefix}_img{image_id:03d}_iter{iteration}_generated.png"
            current_image.save(img_path)

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
                    log_prob_original = compute_caption_log_prob(current_image, original_caption, model, processor)
                    result['log_prob_original_caption'] = log_prob_original
                    print(f"      Log P(original_caption | current_image): {log_prob_original:.3f}")
                except Exception as logprob_error:
                    print(f"      Warning: log-prob computation failed: {logprob_error}")
                finally:
                    clear_memory()

            # Detect convergence based on accumulated vision embeddings
            if added_vision_embedding and vision_embeddings:
                converged, conv_iter, attr_type = detect_convergence(vision_embeddings)
                result['converged'] = converged
                result['attractor_type'] = attr_type
                if conv_iter >= 0:
                    result['convergence_iteration'] = conv_iter + 1
                if converged and iteration < num_iterations - 1:
                    print(f"      CONVERGED to {attr_type} at iteration {result['convergence_iteration']}!")
            
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
                log_prob_caption_on_orig = compute_caption_log_prob(reference_image, caption, model, processor)
                result['log_prob_caption_on_original_image'] = log_prob_caption_on_orig
                print(f"      Log P(caption | original_image): {log_prob_caption_on_orig:.3f}")
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
                print(f"      Generating image via SDXL...")
                seed = (image_id * 100) + (iteration + 1)
                output = replicate.run(
                    SDXL_MODEL_ID,
                    input={"prompt": caption, "seed": seed}
                )
                
                image_url = output[0] if isinstance(output, list) else output
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                current_image = Image.open(response.raw)
                
                clear_memory()
            
        except Exception as e:
            result['error'] = str(e)
            print(f"      Error: {e}")
            # Stop iterations on error
            break
        
        iteration_results.append(result)
    
    return iteration_results

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
    
    # Load Sentence Transformer
    print("\n3. LOADING SENTENCE TRANSFORMER")
    sim_model = SentenceTransformer(SENTIMENT_MODEL, device=SIM_DEVICE)
    print("Sentence transformer loaded")
    
    # Load Dataset
    print("\n4. LOADING DATASET")
    
    dataset_name = "detection-datasets/coco"
    dataset_split = f"val[:{NUM_IMAGES}]"
    try:
        dataset = load_dataset(dataset_name, split=dataset_split)
        print(f"✓ Loaded {len(dataset)} images from COCO dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Prepare prompt
    prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
    
    # Process images
    print("\n5. PROCESSING IMAGES")
    
    all_results = []
    
    for i, sample in enumerate(dataset):
        print(f"\n[{i+1}/{NUM_IMAGES}] Processing image {i+1}...")
        print("-" * 80)
        
        image = sample['image']
        iteration_results = run_multiple_iterations(
            image,
            i + 1,
            model,
            processor,
            sim_model,
            prompt,
            timestamp,
            NUM_ITERATIONS,
            images_dir,
            run_name,
        )
        all_results.extend(iteration_results)
        
        print_gpu_memory()
        print()
    
    # Save results to CSV
    print("\n6. SAVING RESULTS")
    
    csv_path = run_dir / f"{run_name}_iteration_results.csv"
    
    fieldnames = [
        'image_id', 'iteration', 'caption',
        'similarity_to_original', 'similarity_to_previous',
        'bert_f1', 'jaccard', 'length_ratio',
        'vision_embedding_norm', 'vision_sim_to_original', 'vision_sim_to_previous',
        'log_prob_original_caption',
        'log_prob_caption_on_original_image',
        'converged', 'convergence_iteration', 'attractor_type',
        'error'
    ]
    
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"✓ Results saved to: {csv_path}")
    
    # Calculate statistics
    print("\n7. DECAY ANALYSIS")
    
    # Group by iteration
    by_iteration = {}
    for r in all_results:
        if r['similarity_to_original'] is not None and r['error'] is None:
            iter_num = r['iteration']
            if iter_num not in by_iteration:
                by_iteration[iter_num] = []
            by_iteration[iter_num].append(r)
    
    iteration_summary = []
    decay_stats = None
    fastest = []
    most_stable = []
    if by_iteration:
        print("\nAverage Metrics by Iteration:")
        print(f"{'Iter':<6} {'Sim→Orig':<10} {'Sim→Prev':<10} {'BERT-F1':<10} {'Jaccard':<10}")
        
        for iter_num in sorted(by_iteration.keys()):
            results = by_iteration[iter_num]
            avg_sim_orig = np.mean([r['similarity_to_original'] for r in results])
            avg_sim_prev = np.mean([r['similarity_to_previous'] for r in results if r['similarity_to_previous'] is not None])
            avg_bert = np.mean([r['bert_f1'] for r in results if r['bert_f1'] is not None])
            avg_jaccard = np.mean([r['jaccard'] for r in results if r['jaccard'] is not None])
            
            print(f"{iter_num:<6} {avg_sim_orig:<10.3f} {avg_sim_prev:<10.3f} {avg_bert:<10.3f} {avg_jaccard:<10.3f}")
            iteration_summary.append({
                'iteration': iter_num,
                'similarity_to_original_mean': float(avg_sim_orig),
                'similarity_to_previous_mean': float(avg_sim_prev) if not np.isnan(avg_sim_prev) else None,
                'bert_f1_mean': float(avg_bert) if not np.isnan(avg_bert) else None,
                'jaccard_mean': float(avg_jaccard) if not np.isnan(avg_jaccard) else None,
            })
        
        # Calculate decay rate
        if len(by_iteration) >= 2:
            iter1_sim = np.mean([r['similarity_to_original'] for r in by_iteration[1]])
            last_iter = max(by_iteration.keys())
            last_sim = np.mean([r['similarity_to_original'] for r in by_iteration[last_iter]])
            
            decay_rate = (iter1_sim - last_sim) / (last_iter - 1) if last_iter > 1 else 0
            total_decay = iter1_sim - last_sim
            
            print(f"\nDecay Statistics:")
            print(f"  Initial similarity (iter 1): {iter1_sim:.3f}")
            print(f"  Final similarity (iter {last_iter}): {last_sim:.3f}")
            print(f"  Total decay: {total_decay:.3f}")
            print(f"  Average decay per iteration: {decay_rate:.3f}")
            decay_stats = {
                'initial_similarity': float(iter1_sim),
                'final_similarity': float(last_sim),
                'total_decay': float(total_decay),
                'avg_decay_per_iteration': float(decay_rate),
                'last_iteration': int(last_iter),
            }
        
        # Find extreme cases
        print(f"\nFastest Degrading Images (lowest final similarity):")
        final_iter = max(by_iteration.keys())
        final_results = sorted(by_iteration[final_iter], key=lambda x: x['similarity_to_original'])[:5]
        for r in final_results:
            print(f"  Image {r['image_id']}: {r['similarity_to_original']:.3f}")
        fastest = [
            {'image_id': int(r['image_id']), 'similarity_to_original': float(r['similarity_to_original'])}
            for r in final_results
        ]
        
        print(f"\nMost Stable Images (highest final similarity):")
        stable_results = sorted(by_iteration[final_iter], key=lambda x: x['similarity_to_original'], reverse=True)[:5]
        for r in stable_results:
            print(f"  Image {r['image_id']}: {r['similarity_to_original']:.3f}")
        most_stable = [
            {'image_id': int(r['image_id']), 'similarity_to_original': float(r['similarity_to_original'])}
            for r in stable_results
        ]
    
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
            'sdxl_model_id': SDXL_MODEL_ID,
            'convergence_window': CONVERGENCE_WINDOW,
            'convergence_threshold': CONVERGENCE_THRESHOLD,
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
